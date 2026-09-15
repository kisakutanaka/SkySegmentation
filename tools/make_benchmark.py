"""入力・空マスク・合成結果を並べたベンチマーク画像を作る。

    tools/.venv/bin/python tools/make_benchmark.py models/tinyskynet_sky_256.onnx \
        test_img/cloudy01.png test_img/cloudy02.png test_img/IMG_4476.mp4 docs/benchmark.png
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from run_demo import Segmenter, composite, overlay  # noqa: E402

CELL_W = 300          # 1 セルの幅
PAD = 8
HEADER = 96
LABEL = 26
VIDEO_FRAMES = 3      # 動画から抜き出すフレーム数
FONT_PATH = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def sample_video(path, n):
    tmp = tempfile.mkdtemp()
    subprocess.run(['ffmpeg', '-y', '-i', str(path), '-vf', f'scale=-2:720,fps=1/3',
                    f'{tmp}/v_%03d.png', '-loglevel', 'error'], check=True)
    files = sorted(Path(tmp).glob('v_*.png'))
    step = max(1, len(files) // n)
    return [Image.open(f).convert('RGB') for f in files[::step][:n]]


def main():
    model, dst = sys.argv[1], Path(sys.argv[-1])
    seg = Segmenter(model)

    items = []   # (見出し, PIL画像)
    for src in sys.argv[2:-1]:
        p = Path(src)
        if p.suffix.lower() in ('.mp4', '.mov', '.m4v'):
            for i, im in enumerate(sample_video(p, VIDEO_FRAMES)):
                items.append((f'{p.name} #{i + 1}', im))
        else:
            items.append((p.name, Image.open(p).convert('RGB')))

    rows, times = [], []
    for name, im in items:
        t0 = time.perf_counter()
        mask = seg(im)
        times.append((time.perf_counter() - t0) * 1000)
        rows.append((name, im, overlay(im, mask), composite(im, mask, [None, None]),
                     float((mask > 0.5).mean())))

    # セル寸法は最初の画像の縦横比にそろえる（縦横比が違う画像は個別に計算）
    sheet_w = CELL_W * 3 + PAD * 4
    heights = [int(CELL_W * im.height / im.width) for _, im, _, _, _ in rows]
    sheet_h = HEADER + sum(h + LABEL + PAD for h in heights) + PAD

    sheet = Image.new('RGB', (sheet_w, sheet_h), (24, 24, 28))
    d = ImageDraw.Draw(sheet)
    size_kb = Path(model).stat().st_size / 1024
    d.text((PAD, 12), f'Sky segmentation benchmark — {Path(model).name}', font=font(18), fill=(255, 255, 255))
    d.text((PAD, 40), f'{size_kb:,.0f} KB / input {seg.size}x{seg.size} / '
                      f'{np.mean(times):.1f} ms per frame (CPU, 4 threads)',
           font=font(14), fill=(170, 175, 185))
    for i, label in enumerate(('input', 'sky mask', 'composite')):
        d.text((PAD + i * (CELL_W + PAD), 70), label, font=font(14), fill=(120, 200, 255))

    y = HEADER
    for (name, im, ov, comp, ratio), h in zip(rows, heights):
        for i, cell in enumerate((im, ov, comp)):
            sheet.paste(cell.resize((CELL_W, h), Image.LANCZOS), (PAD + i * (CELL_W + PAD), y))
        d.text((PAD, y + h + 6), f'{name}   sky {ratio * 100:.0f}%', font=font(13), fill=(190, 195, 205))
        y += h + LABEL + PAD

    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst)
    print(f'{dst}  ({sheet.size[0]}x{sheet.size[1]}, {dst.stat().st_size/1024:.0f} KB)')
    print(f'inference: {np.mean(times):.1f} ms/frame')


if __name__ == '__main__':
    main()
