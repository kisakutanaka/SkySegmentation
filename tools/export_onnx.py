"""学習した TinySkyNet を ONNX に書き出す（BatchNorm は畳み込みに融合される）。"""
import sys
import torch
from model import TinySkyNet

ckpt = sys.argv[1] if len(sys.argv) > 1 else 'tinyskynet.pt'
out = sys.argv[2] if len(sys.argv) > 2 else 'tinyskynet.onnx'

net = TinySkyNet()
net.load_state_dict(torch.load(ckpt, map_location='cpu'))
net.eval()
torch.onnx.export(
    net, torch.zeros(1, 3, 256, 256), out,
    input_names=['x'], output_names=['logits'],
    opset_version=13, dynamo=False)
print('exported', out)
