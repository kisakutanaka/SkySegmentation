"""SkySeg (U-2-Net, MIT) を教師にして擬似ラベルを作る。

    tools/.venv/bin/python tools/pseudo_label_skyseg.py dataset/skyseg.onnx \
        dataset/images/train dataset/labels_skyseg/train [シャード番号] [シャード数]

出力: <out>/<id>.png  … R=空確率(0-255), G=信頼度(0-255)
PP-MobileSeg 版 (pseudo_label.py) との違い:
  - 教師が空専用の二値モデルなので ADE20K のクラス間マージンが要らない
  - 出力が 320x320 のソフトマットなので、64x64 を起こし直す必要がなく輪郭が正確
  - 教師の迷いは確率そのものが表すため、信頼度は一律 255（ソフトターゲットとして学習させる）
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL, SRC, DST = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
SHARD = int(sys.argv[4]) if len(sys.argv) > 4 else 0
NSHARD = int(sys.argv[5]) if len(sys.argv) > 5 else 1
DST.mkdir(parents=True, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
G = 256   # 擬似ラベルの解像度（PP-MobileSeg 版と同じ）

so = ort.SessionOptions(); so.intra_op_num_threads = 2
sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
name = sess.get_inputs()[0].name

paths = sorted(SRC.glob('*.jpg'))[SHARD::NSHARD]
kept = 0
for i, path in enumerate(paths):
    out = DST / (path.stem + '.png')
    if out.exists():
        kept += 1
        continue
    im = Image.open(path).convert('RGB')
    x = np.asarray(im.resize((320, 320), Image.BILINEAR), np.float32) / 255.
    # U-2-Net は d0..d6 を返す。d0 が本出力で、すでに sigmoid 済み
    prob = sess.run(None, {name: ((x - MEAN) / STD).transpose(2, 0, 1)[None]})[0][0, 0]
    fine = np.asarray(Image.fromarray((prob * 255).astype(np.uint8))
                      .resize((G, G), Image.BILINEAR), np.float32) / 255.

    ratio = float((fine > 0.5).mean())
    if ratio < 0.02 or ratio > 0.95:   # 空が無い/画面がほぼ空 の極端な例は捨てる
        continue
    rgb = np.stack([fine * 255, np.full_like(fine, 255), np.zeros_like(fine)], -1).astype(np.uint8)
    Image.fromarray(rgb).save(out)
    kept += 1
    if (i + 1) % 100 == 0:
        print(f'[{SHARD}] {i+1}/{len(paths)} kept={kept}', flush=True)
print(f'[{SHARD}] done: {kept}/{len(paths)}')
