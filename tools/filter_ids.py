import sys
SKY = '/m/01bqvp'
URBAN = {'/m/0cgh4', '/m/079cl', '/m/01c8br', '/m/01fdzj', '/m/033rq4', '/m/015qff', '/m/03jm5'}
POS = ('1', '1.0')   # Confidence は "1" と "1.0" の両方の表記がある

cur = None
has_sky = False
has_urban = False
out = []
n = 0
for line in sys.stdin:
    p = line.rstrip().split(',')
    if len(p) < 4 or p[0] == 'ImageID':
        continue
    iid, label, conf = p[0], p[2], p[3]
    if iid != cur:
        if cur and has_sky and has_urban:
            out.append(cur)
        cur = iid
        has_sky = has_urban = False
        n += 1
    if conf in POS:
        if label == SKY:
            has_sky = True
        elif label in URBAN:
            has_urban = True
if cur and has_sky and has_urban:
    out.append(cur)
sys.stderr.write(f'scanned {n} images, matched {len(out)}\n')
print('\n'.join(out))
