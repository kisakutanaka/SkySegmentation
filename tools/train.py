"""擬似ラベルから TinySkyNet を蒸留学習する。"""
import random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import TinySkyNet

IMG = 256
OUT = 128
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SkyData(Dataset):
    def __init__(self, img_dir, lab_dir, train=True):
        self.lab = sorted(Path(lab_dir).glob('*.png'))
        self.img_dir = Path(img_dir)
        self.train = train

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        lp = self.lab[i]
        im = Image.open(self.img_dir / (lp.stem + '.jpg')).convert('RGB')
        lab = Image.open(lp)
        if self.train:
            # ランダムな切り出しと回転で、手持ち撮影の多様な構図に備える
            if random.random() < 0.5:
                im, lab = im.transpose(Image.FLIP_LEFT_RIGHT), lab.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.3:
                ang = random.uniform(-15, 15)
                im = im.rotate(ang, Image.BILINEAR, expand=False)
                lab = lab.rotate(ang, Image.BILINEAR, expand=False)
            s = random.uniform(0.6, 1.0)
            w, h = im.size
            cw, ch = int(w * s), int(h * s)
            x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
            im = im.crop((x0, y0, x0 + cw, y0 + ch))
            lab = lab.crop((int(x0 / w * lab.width), int(y0 / h * lab.height),
                            int((x0 + cw) / w * lab.width), int((y0 + ch) / h * lab.height)))
        im = im.resize((IMG, IMG), Image.BILINEAR)
        lab = lab.resize((OUT, OUT), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, np.float32).transpose(2, 0, 1) / 255.)
        if self.train and random.random() < 0.5:   # 明るさ・コントラストの揺らぎ
            x = torch.clamp(x * random.uniform(0.7, 1.3) + random.uniform(-0.1, 0.1), 0, 1)
        x = (x - MEAN) / STD
        l = np.asarray(lab, np.float32) / 255.
        return x, torch.from_numpy(l[:, :, 0]), torch.from_numpy(l[:, :, 1])


def save(obj, path):
    """書き込み中に止められてもファイルが壊れないよう、別名で書いてから差し替える"""
    tmp = str(path) + '.tmp'
    torch.save(obj, tmp)
    Path(tmp).replace(path)


def iou(pred, target):
    p, t = pred > 0.5, target > 0.5
    inter = (p & t).sum()
    union = (p | t).sum()
    return float(inter) / float(union) if union else 1.0


def main():
    # 使い方: train.py [エポック数] [ラベルのディレクトリ] [チェックポイントの保存先]
    #   例) train.py 40 dataset/labels_skyseg dataset/tinyskynet_skyseg.pt
    #
    # 毎エポック <保存先>.last に途中経過（optimizer と scheduler を含む）を書くので、
    # 止めたあと同じコマンドを打てば続きから再開する。
    # <保存先> 本体には val IoU が最良のときだけ重みを書き出す（export_onnx.py はこちらを読む）。
    lab_root = sys.argv[2] if len(sys.argv) > 2 else 'dataset/labels'
    ckpt = sys.argv[3] if len(sys.argv) > 3 else 'dataset/tinyskynet.pt'
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tr = DataLoader(SkyData('dataset/images/train', f'{lab_root}/train', True), batch_size=16,
                    shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    va = DataLoader(SkyData('dataset/images/val', f'{lab_root}/val', False), batch_size=16, num_workers=2)
    print(f'train {len(tr.dataset)} / val {len(va.dataset)} images, device={dev}, labels={lab_root}')

    net = TinySkyNet().to(dev)
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, epochs=epochs, steps_per_epoch=len(tr))
    best = 0.0
    start = 0
    last = ckpt + '.last'
    if Path(last).exists():   # 中断した学習の再開
        st = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched']); best = st['best']; start = st['ep']
        print(f'resume from {last}: epoch {start + 1} から / best {best:.4f}', flush=True)

    for ep in range(start, epochs):
        net.train(); t0 = time.time(); tot = 0.0
        for x, y, c in tr:
            x, y, c = x.to(dev), y.to(dev), c.to(dev)
            logit = net(x)[:, 0]
            # 教師が迷っている画素(c が小さい)は損失から外す
            bce = F.binary_cross_entropy_with_logits(logit, y, reduction='none')
            loss = (bce * c).sum() / c.sum().clamp(min=1)
            p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * y * c).sum() + 1) / ((p * c).sum() + (y * c).sum() + 1)
            loss = loss + 0.5 * dice
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        net.eval(); ious = []
        with torch.no_grad():
            for x, y, c in va:
                p = torch.sigmoid(net(x.to(dev))[:, 0]).cpu()
                ious += [iou(p[i].numpy(), y[i].numpy()) for i in range(len(p))]
        m = float(np.mean(ious))
        print(f'ep {ep+1}/{epochs} loss {tot/len(tr):.4f} val IoU(vs teacher) {m:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        if m > best:
            best = m
            save(net.state_dict(), ckpt)
        # 途中経過は毎エポック上書きする（ここで止めても次回は続きから）
        save({'ep': ep + 1, 'best': best, 'model': net.state_dict(),
              'opt': opt.state_dict(), 'sched': sched.state_dict()}, last)
    print('best val IoU:', round(best, 4))


if __name__ == '__main__':
    main()
