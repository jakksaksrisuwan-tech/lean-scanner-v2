import numpy as np, json, glob, os
from PIL import Image, ImageDraw
D = '/Users/jsaksrisuwan/workspace/lean_scanner_v2/captures-debug'
files = sorted(glob.glob(os.path.join(D, 'dump_20260720_183*_480x640.json')))
imgs = []
for fp in files:
    j = json.loads(open(fp).read())
    base = fp[:-5]
    rgb = np.frombuffer(open(base + '.rgb.raw', 'rb').read(), dtype=np.uint8).reshape(j['h'], j['w'], 3)
    img = Image.fromarray(rgb[:, :, ::-1].copy())
    d = ImageDraw.Draw(img)
    q = j.get('rawQuad')
    if q:
        for k in range(4):
            d.line([(q[k*2], q[k*2+1]), (q[(k+1)%4*2], q[(k+1)%4*2+1])], fill='cyan', width=3)
    sq = j.get('smoothQuad')
    if sq:
        for k in range(4):
            d.line([(sq[k*2], sq[k*2+1]), (sq[(k+1)%4*2], sq[(k+1)%4*2+1])], fill='yellow', width=2)
    # green = detection OK, red = null
    found = q is not None
    color = 'lime' if found else 'red'
    d.text((4, 4), ('DET' if found else 'NULL'), fill=color)
    imgs.append((os.path.basename(fp), img))
W = 240 * 3 + 12; H = 320 * ((len(imgs)+2)//3)
combo = Image.new('RGB', (W, H), 'black')
for i, (name, img) in enumerate(imgs):
    r, c = divmod(i, 3)
    th = img.resize((240, 320))
    combo.paste(th, (c * 240, r * 320))
    d = ImageDraw.Draw(combo)
    d.text((c * 240 + 4, r * 320 + 4), name[:24], fill='white')
combo.save('/tmp/dumps_NEWBATCH.png')
print('saved /tmp/dumps_NEWBATCH.png  ' + str(len(imgs)) + ' frames')
