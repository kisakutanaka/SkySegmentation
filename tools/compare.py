"""教師(PP-MobileSeg-Base) と 生徒(TinySkyNet) を並べて比較する。"""
import sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)

def box(a, r):
    c=np.cumsum(np.pad(a,((1,0),(0,0))),0); d=np.arange(a.shape[0])
    lo=np.maximum(d-r,0); hi=np.minimum(d+r+1,a.shape[0]); a=(c[hi]-c[lo])/(hi-lo)[:,None]
    c=np.cumsum(np.pad(a,((0,0),(1,0))),1); d=np.arange(a.shape[1])
    lo=np.maximum(d-r,0); hi=np.minimum(d+r+1,a.shape[1]); return (c[:,hi]-c[:,lo])/(hi-lo)[None,:]

def guided(I,p,r=8,eps=1e-4):
    mI,mp=box(I,r),box(p,r)
    a=(box(I*p,r)-mI*mp)/((box(I*I,r)-mI*mI)+eps)
    return np.clip(box(a,r)*I+box(mp-a*mI,r),0,1)

so = ort.SessionOptions(); so.intra_op_num_threads = 1
teacher = ort.InferenceSession(sys.argv[1], so, providers=['CPUExecutionProvider'])
student = ort.InferenceSession(sys.argv[2], so, providers=['CPUExecutionProvider'])

def run_teacher(im):
    x = np.asarray(im.resize((512,512), Image.BILINEAR), np.float32)/255.
    t=time.perf_counter()
    L = teacher.run(None, {teacher.get_inputs()[0].name: ((x-MEAN)/STD).transpose(2,0,1)[None]})[0][0]
    ms=(time.perf_counter()-t)*1000
    d = L[2]-np.max(np.delete(L,2,0),0)
    return 1/(1+np.exp(-(d-2))), ms

def run_student(im):
    x = np.asarray(im.resize((256,256), Image.BILINEAR), np.float32)/255.
    t=time.perf_counter()
    L = student.run(None, {student.get_inputs()[0].name: ((x-MEAN)/STD).transpose(2,0,1)[None]})[0][0,0]
    ms=(time.perf_counter()-t)*1000
    return 1/(1+np.exp(-L)), ms

def refine(im, p, G=256):
    g = np.asarray(im.convert('L').resize((G,G), Image.BILINEAR), np.float32)/255.
    up = np.asarray(Image.fromarray((p*255).astype(np.uint8)).resize((G,G), Image.BILINEAR), np.float32)/255.
    return guided(g, up)

paths = sorted(Path(sys.argv[3]).glob('*.jpg'))[:int(sys.argv[4]) if len(sys.argv)>4 else 8]
W=H=220
sheet = Image.new('RGB',(W*3, H*len(paths)))
tms=sms=[]; tms=[]; sms=[]; ious=[]
for i,path in enumerate(paths):
    im = Image.open(path).convert('RGB')
    tp, t1 = run_teacher(im); sp, t2 = run_student(im)
    tms.append(t1); sms.append(t2)
    tf, sf = refine(im, tp), refine(im, sp)
    inter=((tf>0.5)&(sf>0.5)).sum(); union=((tf>0.5)|(sf>0.5)).sum()
    ious.append(inter/union if union else 1.0)
    base = im.resize((W,H))
    sheet.paste(base, (0, i*H))
    for j,m in ((1,tf),(2,sf)):
        mk = Image.fromarray((m*255).astype(np.uint8)).resize((W,H))
        sheet.paste(Image.composite(Image.new('RGB',(W,H),(255,0,0)), base, mk.point(lambda v:int(v*0.6))), (j*W, i*H))
sheet.save(sys.argv[5] if len(sys.argv)>5 else 'compare.png')
print(f'teacher {np.mean(tms):.0f} ms / student {np.mean(sms):.1f} ms  ({np.mean(tms)/np.mean(sms):.0f}x faster)')
print(f'IoU(student vs teacher): {np.mean(ious):.3f}')
print('左=元画像 中=教師 右=生徒')
