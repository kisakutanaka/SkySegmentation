"""同梱モデルで画像／動画を処理し、マスクと合成結果を書き出す。
ブラウザ側 (sky-segmenter.js + app.js) と同じ手順を Python で再現している。

    tools/.venv/bin/python tools/run_demo.py models/tinyskynet_sky_256.onnx test_img/foo.png out/
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
REFINE = 512      # マスク／ガイドの解像度（sky-segmenter.js の refineSize）
RADIUS = 4        # ガイデッドフィルタの半径（モデル入力解像度上の画素数）
EPS = 1e-4
SHARPEN = 6       # 確率の 0→1 遷移をどれだけ立てるか
INERTIA = 0.6     # 動画でのマスクの時間平滑化
KAIJU = Path(__file__).resolve().parent.parent / 'kaiju.png'
KAIJU_SCALE = 0.45


def box(a, r):
    c = np.cumsum(np.pad(a, ((1, 0), (0, 0))), 0); d = np.arange(a.shape[0])
    lo = np.maximum(d - r, 0); hi = np.minimum(d + r + 1, a.shape[0])
    a = (c[hi] - c[lo]) / (hi - lo)[:, None]
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), 1); d = np.arange(a.shape[1])
    lo = np.maximum(d - r, 0); hi = np.minimum(d + r + 1, a.shape[1])
    return (c[:, hi] - c[:, lo]) / (hi - lo)[None, :]


def guided_coeffs(I, p, r=RADIUS, eps=EPS):
    """ガイデッドフィルタの線形係数 a, b（マスク ≒ a * ガイド + b）"""
    mI, mp = box(I, r), box(p, r)
    a = (box(I * p, r) - mI * mp) / ((box(I * I, r) - mI * mI) + eps)
    return box(a, r), box(mp - a * mI, r)


def resize(a, size):
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), np.float32)


class Segmenter:
    def __init__(self, model):
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(model, so, providers=['CPUExecutionProvider'])
        self.name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = shape[2] if isinstance(shape[2], int) else 512

    def __call__(self, im):
        size, cap = self.size, max(self.size, REFINE)
        pool = max(1, round(cap / size))

        # 1. マスク解像度で取り込み、面積平均でモデル入力サイズへ落とす
        hi = np.asarray(im.convert('RGB').resize((cap, cap), Image.BILINEAR), np.float32)
        lo = hi.reshape(size, pool, size, pool, 3).mean((1, 3))

        # 2. 推論 → 空である確率（低解像度）
        x = ((lo / 255. - MEAN) / STD).transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.name: x})[0][0]
        if out.shape[0] == 1:                      # 二値モデル
            prob = 1 / (1 + np.exp(-out[0]))
        else:                                      # ADE20K 150 クラスモデル
            d = out[2] - np.max(np.delete(out, 2, 0), 0)
            prob = 1 / (1 + np.exp(-(d - 2.0)))

        # 3. fast guided filter: 係数はモデル入力解像度で求め、高解像度のガイドに当てる
        w = np.array([0.299, 0.587, 0.114], np.float32)
        gl, gh = (lo @ w) / 255., (hi @ w) / 255.
        a, b = guided_coeffs(gl, resize((prob * 255).astype(np.uint8), size) / 255.)
        mask = resize(a.astype(np.float32), cap) * gh + resize(b.astype(np.float32), cap)
        return np.clip((mask - 0.5) * SHARPEN + 0.5, 0, 1)


def composite(im, mask):
    """後景(元画像) → 怪獣 → 前景(空以外) の順に重ねる（app.js と同じ）"""
    w, h = im.size
    out = im.convert('RGB').copy()
    kaiju = Image.open(KAIJU).convert('RGBA')
    kw = min(w, h) * KAIJU_SCALE
    kh = kw * kaiju.height / kaiju.width
    kaiju = kaiju.resize((int(kw), int(kh)), Image.LANCZOS)
    out.paste(kaiju, (int(w / 2 - kw / 2), int(h / 2 - kh / 2)), kaiju)
    alpha = Image.fromarray(((1 - mask) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    out.paste(im.convert('RGB'), (0, 0), alpha)     # 前景を最前面に戻す
    return out


def overlay(im, mask, color=(0, 229, 255)):
    w, h = im.size
    m = Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return Image.composite(Image.new('RGB', (w, h), color), im.convert('RGB'),
                           m.point(lambda v: int(v * 0.55)))


def run_image(seg, path, out_dir):
    im = Image.open(path).convert('RGB')
    mask = seg(im)
    w, h = im.size
    sheet = Image.new('RGB', (w * 3, h))
    sheet.paste(im, (0, 0))
    sheet.paste(overlay(im, mask), (w, 0))
    sheet.paste(composite(im, mask), (w * 2, 0))
    sheet.thumbnail((1800, 1800), Image.LANCZOS)
    dst = out_dir / f'{Path(path).stem}_result.png'
    sheet.save(dst)
    print(f'{Path(path).name}: sky {float((mask > 0.5).mean()):.2f} -> {dst}')


def run_video(seg, path, out_dir, fps=30):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(['ffmpeg', '-y', '-i', str(path), '-vf', 'scale=-2:720',
                        str(tmp / 'f_%05d.png'), '-loglevel', 'error'], check=True)
        frames = sorted(tmp.glob('f_*.png'))
        print(f'{Path(path).name}: {len(frames)} frames')
        smoothed = None
        (tmp / 'out').mkdir()
        for i, f in enumerate(frames):
            im = Image.open(f).convert('RGB')
            m = seg(im)
            smoothed = m if smoothed is None else smoothed * INERTIA + m * (1 - INERTIA)
            side = Image.new('RGB', (im.width * 2, im.height))
            side.paste(overlay(im, smoothed), (0, 0))
            side.paste(composite(im, smoothed), (im.width, 0))
            side.save(tmp / 'out' / f.name)
            if (i + 1) % 60 == 0:
                print(f'  {i+1}/{len(frames)}', flush=True)
        dst = out_dir / f'{Path(path).stem}_result.mp4'
        subprocess.run(['ffmpeg', '-y', '-framerate', str(fps), '-i', str(tmp / 'out' / 'f_%05d.png'),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', str(dst),
                        '-loglevel', 'error'], check=True)
        print(f'-> {dst}')


if __name__ == '__main__':
    model, out_dir = sys.argv[1], Path(sys.argv[-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = Segmenter(model)
    for src in sys.argv[2:-1]:
        if Path(src).suffix.lower() in ('.mp4', '.mov', '.m4v'):
            run_video(seg, src, out_dir)
        else:
            run_image(seg, src, out_dir)
