"""現行パイプライン（PP-MobileSeg-Base + margin + guided filter）で擬似ラベルを作る。

出力: <out>/<id>.png  … R=空確率(0-255), G=信頼度(0-255)
信頼度が低い画素は学習時に損失から除外する（教師が迷っている場所を生徒に教えない）。
"""
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL = sys.argv[1]
SRC = Path(sys.argv[2])
DST = Path(sys.argv[3])
DST.mkdir(parents=True, exist_ok=True)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MARGIN = 2.0
G = 256  # 擬似ラベルの解像度

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

so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(MODEL, so, providers=['CPUExecutionProvider'])
name = sess.get_inputs()[0].name

paths = sorted(SRC.glob('*.jpg'))
kept = 0
for i, path in enumerate(paths):
    out = DST / (path.stem + '.png')
    if out.exists():
        kept += 1
        continue
    im = Image.open(path).convert('RGB')
    x = np.asarray(im.resize((512, 512), Image.BILINEAR), np.float32) / 255.
    L = sess.run(None, {name: ((x - MEAN) / STD).transpose(2, 0, 1)[None]})[0][0]
    sky = L[2]
    other = np.max(np.delete(L, 2, 0), 0)
    diff = sky - other
    prob = 1 / (1 + np.exp(-(diff - MARGIN)))
    # 教師の確信度: |sky - other| が大きいほど信頼できる
    conf = np.clip(np.abs(diff - MARGIN) / 4.0, 0, 1)

    guide = np.asarray(im.convert('L').resize((G, G), Image.BILINEAR), np.float32) / 255.
    up = np.asarray(Image.fromarray((prob * 255).astype(np.uint8)).resize((G, G), Image.BILINEAR), np.float32) / 255.
    fine = guided(guide, up)
    conf_up = np.asarray(Image.fromarray((conf * 255).astype(np.uint8)).resize((G, G), Image.BILINEAR), np.float32) / 255.

    ratio = float((fine > 0.5).mean())
    if ratio < 0.02 or ratio > 0.95:   # 空が無い/画面がほぼ空 の極端な例は捨てる
        continue
    rgb = np.stack([(fine * 255), (conf_up * 255), np.zeros_like(fine)], -1).astype(np.uint8)
    Image.fromarray(rgb).save(out)
    kept += 1
    if (i + 1) % 200 == 0:
        print(f'{i+1}/{len(paths)} kept={kept}', flush=True)
print(f'done: {kept}/{len(paths)}')
