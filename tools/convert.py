#!/usr/bin/env python3
"""PP-MobileSeg-Tiny (PaddleSeg / Apache-2.0) を、このサンプル用の ONNX に変換する。

生成物 models/pp_mobileseg_tiny_ade20k_512.onnx はリポジトリに同梱済みなので、
このスクリプトを実行する必要は普段ありません。再現性のために置いてあります。

    uv venv --python 3.12 tools/.venv
    uv pip install --python tools/.venv/bin/python \
        paddlepaddle paddle2onnx onnx onnxruntime onnxslim pillow numpy \
        pyyaml opencv-python-headless scipy prettytable filelock six requests \
        tqdm scikit-learn scikit-image setuptools
    tools/.venv/bin/python tools/convert.py

やっていること:
  1. PaddleSeg 本体（config が必要）と学習済み重みを取得
  2. F.interpolate にパッチを当てる（後述）
  3. 最終アップサンプルを含まない形でエクスポート
     → 出力が [1,150,32,32] の低解像度ロジットになり、ブラウザ側で扱える大きさになる
       （素直に出すと [1,150,512,512] = float32 で 157MB になり毎フレーム読み出せない）
  4. paddle2onnx → onnxslim
  5. 実写で検証
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "tools" / "_work"
OUT = ROOT / "models" / "pp_mobileseg_tiny_ade20k_512.onnx"
CONFIG = "PaddleSeg/configs/pp_mobileseg/pp_mobileseg_tiny_ade20k_512x512_80k.yml"
WEIGHTS_URL = "https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_tiny/model.pdparams"
INPUT_SIZE = 512


def fetch():
    WORK.mkdir(parents=True, exist_ok=True)
    if not (WORK / "PaddleSeg").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/PaddlePaddle/PaddleSeg.git"],
            cwd=WORK, check=True)
    weights = WORK / "model.pdparams"
    if not weights.exists():
        subprocess.run(["curl", "-L", "-o", str(weights), WEIGHTS_URL], check=True)
    return weights


def patch_interpolate():
    """paddle2onnx の PIR パーサは interpolate の `scale` 属性を float 配列として読むが、
    paddle 3.x は double で格納するため、scale_factor 指定の補間があると変換が落ちる
    （AAM 内の `F.interpolate(x, scale_factor=0.5)` が該当）。
    入力サイズは静的に決まるので、等価な size 指定に書き換えて回避する。"""
    import paddle
    import paddle.nn.functional as F

    original = F.interpolate

    @paddle.jit.not_to_static
    def patched(x, size=None, scale_factor=None, **kw):
        if scale_factor is not None and size is None:
            in_shape = x.shape[2:]
            factors = scale_factor if isinstance(scale_factor, (list, tuple)) \
                else [scale_factor] * len(in_shape)
            if all(isinstance(i, int) and i > 0 for i in in_shape):
                size = [int(round(i * f)) for i, f in zip(in_shape, factors)]
                return original(x, size=size, **kw)
        return original(x, size=size, scale_factor=scale_factor, **kw)

    F.interpolate = patched
    import paddleseg.models.backbones.strideformer as strideformer
    import paddleseg.models.pp_mobileseg as pp_mobileseg
    strideformer.F.interpolate = patched
    pp_mobileseg.F.interpolate = patched


def export(weights):
    import paddle
    from paddleseg.cvlibs import Config, SegBuilder
    from paddleseg.utils import utils

    patch_interpolate()
    model = SegBuilder(Config(str(WORK / CONFIG))).model
    utils.load_entire_model(model, str(weights))
    model.eval()

    class LowResLogits(paddle.nn.Layer):
        """最終アップサンプルを省き、デコードヘッドの低解像度ロジットをそのまま返す。
        拡大はブラウザ側の canvas 拡大で行うので、ここで戻す情報は無い。"""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.decode_head(self.m.backbone(x))

    wrapped = LowResLogits(model)
    wrapped.eval()
    with paddle.no_grad():
        print("low-res logits:", wrapped(paddle.randn([1, 3, INPUT_SIZE, INPUT_SIZE])).shape)

    save_dir = WORK / "exported"
    paddle.jit.save(wrapped, str(save_dir / "model"), input_spec=[
        paddle.static.InputSpec([1, 3, INPUT_SIZE, INPUT_SIZE], "float32", "x")])
    return save_dir


def to_onnx(save_dir):
    raw = WORK / "raw.onnx"
    subprocess.run([
        sys.executable, "-m", "paddle2onnx",
        "--model_dir", str(save_dir),
        "--model_filename", "model.json",
        "--params_filename", "model.pdiparams",
        "--save_file", str(raw),
        "--opset_version", "13",
    ], check=True)
    subprocess.run([sys.executable, "-m", "onnxslim", str(raw), str(OUT)], check=True)
    print("saved", OUT, OUT.stat().st_size, "bytes")


def validate(image=None):
    """空が写った画像で推論し、クラス 2 (sky) が実際に空を捉えているか確認する。"""
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    if image is None:
        image = WORK / "sample.jpg"
        if not image.exists():
            subprocess.run(["curl", "-L", "-o", str(image),
                            "https://picsum.photos/id/1018/800/600"], check=True)
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    im = Image.open(image).convert("RGB")
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = np.asarray(im.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR), np.float32) / 255.
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    sky = logits[2]
    other = np.max(np.delete(logits, 2, axis=0), axis=0)
    prob = 1 / (1 + np.exp(other - sky))
    print(f"logits {logits.shape} / sky ratio {float((prob > 0.5).mean()):.2f}")
    mask = Image.fromarray((prob * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)
    overlay = WORK / "validation.png"
    Image.composite(Image.new("RGB", im.size, (255, 0, 0)), im,
                    mask.point(lambda v: int(v * 0.6))).save(overlay)
    print("overlay written to", overlay)


if __name__ == "__main__":
    to_onnx(export(fetch()))
    validate()
