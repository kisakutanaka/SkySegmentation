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
from PIL import Image, ImageDraw

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
REFINE = 256      # エッジ吸着を行う解像度
INERTIA = 0.6     # 動画でのマスクの時間平滑化


def box(a, r):
    c = np.cumsum(np.pad(a, ((1, 0), (0, 0))), 0); d = np.arange(a.shape[0])
    lo = np.maximum(d - r, 0); hi = np.minimum(d + r + 1, a.shape[0])
    a = (c[hi] - c[lo]) / (hi - lo)[:, None]
    c = np.cumsum(np.pad(a, ((0, 0), (1, 0))), 1); d = np.arange(a.shape[1])
    lo = np.maximum(d - r, 0); hi = np.minimum(d + r + 1, a.shape[1])
    return (c[:, hi] - c[:, lo]) / (hi - lo)[None, :]


def guided(I, p, r=8, eps=1e-4):
    mI, mp = box(I, r), box(p, r)
    a = (box(I * p, r) - mI * mp) / ((box(I * I, r) - mI * mI) + eps)
    return np.clip(box(a, r) * I + box(mp - a * mI, r), 0, 1)


class Segmenter:
    def __init__(self, model):
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(model, so, providers=['CPUExecutionProvider'])
        self.name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = shape[2] if isinstance(shape[2], int) else 512

    def __call__(self, im):
        x = np.asarray(im.resize((self.size, self.size), Image.BILINEAR), np.float32) / 255.
        out = self.sess.run(None, {self.name: ((x - MEAN) / STD).transpose(2, 0, 1)[None]})[0][0]
        if out.shape[0] == 1:                      # 二値モデル
            prob = 1 / (1 + np.exp(-out[0]))
        else:                                      # ADE20K 150 クラスモデル
            d = out[2] - np.max(np.delete(out, 2, 0), 0)
            prob = 1 / (1 + np.exp(-(d - 2.0)))
        guide = np.asarray(im.convert('L').resize((REFINE, REFINE), Image.BILINEAR), np.float32) / 255.
        up = np.asarray(Image.fromarray((prob * 255).astype(np.uint8))
                        .resize((REFINE, REFINE), Image.BILINEAR), np.float32) / 255.
        return guided(guide, up)


def draw_smiley(img, cx, cy, size):
    """絵文字フォントに依存しないよう図形で描く"""
    d = ImageDraw.Draw(img)
    r = size / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 205, 60), outline=(230, 170, 30), width=max(2, int(r * 0.05)))
    er = r * 0.13
    for ex in (cx - r * 0.35, cx + r * 0.35):
        d.ellipse([ex - er, cy - r * 0.3 - er * 1.4, ex + er, cy - r * 0.3 + er * 1.4], fill=(60, 40, 20))
    d.arc([cx - r * 0.55, cy - r * 0.2, cx + r * 0.55, cy + r * 0.6], 20, 160,
          fill=(60, 40, 20), width=max(3, int(r * 0.11)))


def composite(im, mask, smoothed_center):
    """後景(元画像) → スマイリー → 前景(空以外) の順に重ねる"""
    w, h = im.size
    ys, xs = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    total = mask.sum()
    if total > mask.size * 0.02:
        cx, cy = float((xs * mask).sum() / total / mask.shape[1]), float((ys * mask).sum() / total / mask.shape[0])
        if smoothed_center[0] is None:
            smoothed_center[:] = [cx, cy]
        else:
            smoothed_center[0] += (cx - smoothed_center[0]) * 0.2
            smoothed_center[1] += (cy - smoothed_center[1]) * 0.2
    cx, cy = smoothed_center if smoothed_center[0] is not None else (0.5, 0.3)

    out = im.convert('RGB').copy()
    draw_smiley(out, cx * w, cy * h, min(w, h) * 0.28)
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
    sheet.paste(composite(im, mask, [None, None]), (w * 2, 0))
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
        center = [None, None]
        (tmp / 'out').mkdir()
        for i, f in enumerate(frames):
            im = Image.open(f).convert('RGB')
            m = seg(im)
            smoothed = m if smoothed is None else smoothed * INERTIA + m * (1 - INERTIA)
            side = Image.new('RGB', (im.width * 2, im.height))
            side.paste(overlay(im, smoothed), (0, 0))
            side.paste(composite(im, smoothed, center), (im.width, 0))
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
