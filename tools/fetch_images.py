"""Open Images の画像を取得して 640px に縮小して保存する（元画像は保持しない）"""
import random, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from PIL import Image

split, id_file, n, out_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)
ids = [l.strip() for l in open(id_file) if l.strip()]
random.seed(0)
random.shuffle(ids)
ids = ids[:n]

def fetch(iid):
    dst = out_dir / f'{iid}.jpg'
    if dst.exists():
        return True
    try:
        url = f'https://open-images-dataset.s3.amazonaws.com/{split}/{iid}.jpg'
        with urllib.request.urlopen(url, timeout=30) as r:
            im = Image.open(BytesIO(r.read()))
        im.draft('RGB', (640, 640))   # JPEG を縮小しながらデコード（大幅に速い）
        im = im.convert('RGB')
        im.thumbnail((640, 640), Image.BILINEAR)
        im.save(dst, quality=88)
        return True
    except Exception:
        return False

with ThreadPoolExecutor(max_workers=32) as ex:
    ok = sum(ex.map(fetch, ids))
print(f'{ok}/{len(ids)} downloaded into {out_dir}')
