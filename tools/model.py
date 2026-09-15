"""空/非空の二値セグメンテーション専用の小型 UNet。

150 クラスを判別する必要がないので、パラメータは 1/40 以下で済む。
入力 256x256 / 出力 128x128 のロジット。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dsconv(cin, cout, stride=1):
    """depthwise separable conv + BN + ReLU"""
    return nn.Sequential(
        nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
        nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class TinySkyNet(nn.Module):
    def __init__(self, w=(16, 32, 64, 96)):
        super().__init__()
        c1, c2, c3, c4 = w
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, 2, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(dsconv(c1, c1), dsconv(c1, c2, 2))
        self.enc2 = nn.Sequential(dsconv(c2, c2), dsconv(c2, c3, 2))
        self.enc3 = nn.Sequential(dsconv(c3, c3), dsconv(c3, c4, 2))
        self.mid = dsconv(c4, c4)
        self.dec3 = nn.Sequential(nn.Conv2d(c4 + c3, c3, 1, bias=False),
                                  nn.BatchNorm2d(c3), nn.ReLU(inplace=True), dsconv(c3, c3))
        self.dec2 = nn.Sequential(nn.Conv2d(c3 + c2, c2, 1, bias=False),
                                  nn.BatchNorm2d(c2), nn.ReLU(inplace=True), dsconv(c2, c2))
        self.dec1 = nn.Sequential(nn.Conv2d(c2 + c1, c1, 1, bias=False),
                                  nn.BatchNorm2d(c1), nn.ReLU(inplace=True), dsconv(c1, c1))
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        s = self.stem(x)      # 1/2
        e1 = self.enc1(s)     # 1/4
        e2 = self.enc2(e1)    # 1/8
        e3 = self.enc3(e2)    # 1/16
        y = self.mid(e3)
        y = F.interpolate(y, scale_factor=2.0, mode='bilinear', align_corners=False)
        y = self.dec3(torch.cat([y, e2], 1))   # 1/8
        y = F.interpolate(y, scale_factor=2.0, mode='bilinear', align_corners=False)
        y = self.dec2(torch.cat([y, e1], 1))   # 1/4
        y = F.interpolate(y, scale_factor=2.0, mode='bilinear', align_corners=False)
        y = self.dec1(torch.cat([y, s], 1))    # 1/2
        return self.head(y)                    # 128x128 logits


if __name__ == '__main__':
    m = TinySkyNet()
    n = sum(p.numel() for p in m.parameters())
    print(f'params: {n:,} ({n*4/1e6:.2f} MB fp32)')
    print('out:', m(torch.zeros(1, 3, 256, 256)).shape)
