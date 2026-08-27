#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
import csv, hashlib, random, shutil
import cv2

ROOT=Path('/home/win98/my_projects/yoloproj/armorproj_v2')
OLD=Path('/home/win98/my_projects/yoloproj/armorproj/datasets/armor')
HK=ROOT/'external/hkust_data/armor/RM2025-Armor-Public-Dataset'
CAR=ROOT/'external/hkust_data/car/RM2025-Car-Public-Dataset'
OUT=ROOT/'datasets/armor_det'
RNG=random.Random(260827)

def split_for(group):
    n=int(hashlib.sha1(group.encode()).hexdigest()[:8],16)%100
    return 'train' if n<80 else ('val' if n<90 else 'test')

def add_record(records, source, img, lab, group, negative=False):
    records.append(dict(source=source,img=img,lab=lab,group=group,split=split_for(group),negative=negative))

records=[]
for split in ('train','val'):
    for img in sorted((OLD/'images'/split).glob('*')):
        if img.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
        lab=OLD/'labels'/split/(img.stem+'.txt')
        batch=img.stem.rsplit('_batch',1)[-1] if '_batch' in img.stem else img.stem
        add_record(records,'ansidd',img,lab,'old_batch_'+batch)

for img in sorted(HK.glob('*.jpg')):
    try: block=int(img.stem)//50
    except: block=img.stem[:4]
    add_record(records,'hkust_armor',img,img.with_suffix('.txt'),f'hkust_block_{block}')

# Extract real background crops that do not overlap expanded robot boxes.
negdir=ROOT/'generated/negative_backgrounds'; negdir.mkdir(parents=True,exist_ok=True)
made=0
for img in sorted(CAR.glob('*.jpg')):
    if made>=1000: break
    im=cv2.imread(str(img))
    if im is None: continue
    h,w=im.shape[:2]; lab=img.with_suffix('.txt')
    boxes=[]
    if lab.exists():
        for line in lab.read_text().splitlines():
            a=line.split()
            if len(a)<5: continue
            xc,yc,bw,bh=map(float,a[1:5]); x1=(xc-bw/2)*w; y1=(yc-bh/2)*h; x2=(xc+bw/2)*w; y2=(yc+bh/2)*h
            pad=.12*max(x2-x1,y2-y1); boxes.append((x1-pad,y1-pad,x2+pad,y2+pad))
    side=max(128,int(min(h,w)*RNG.uniform(.28,.48)))
    if side>=min(h,w): continue
    chosen=None
    for _ in range(40):
        x=RNG.randint(0,w-side); y=RNG.randint(0,h-side); rx2=x+side; ry2=y+side
        overlap=False
        for x1,y1,x2,y2 in boxes:
            if max(0,min(rx2,x2)-max(x,x1))*max(0,min(ry2,y2)-max(y,y1))>0:
                overlap=True; break
        if not overlap: chosen=(x,y,side); break
    if chosen is None: continue
    x,y,side=chosen; crop=im[y:y+side,x:x+side]
    dst=negdir/f'carbg_{img.stem}.jpg'; cv2.imwrite(str(dst),crop,[cv2.IMWRITE_JPEG_QUALITY,92]); dst.with_suffix('.txt').write_text('')
    try: block=int(img.stem)//50
    except: block=img.stem[:4]
    add_record(records,'hkust_car_background',dst,dst.with_suffix('.txt'),f'car_block_{block}',True); made+=1

# Exact-content deduplication, preserving first occurrence.
seen=set(); unique=[]
for r in records:
    md5=hashlib.md5(r['img'].read_bytes()).hexdigest()
    if md5 in seen: continue
    seen.add(md5); r['md5']=md5; unique.append(r)
records=unique

for split in ('train','val','test'):
    (OUT/'images'/split).mkdir(parents=True,exist_ok=True)
    (OUT/'labels'/split).mkdir(parents=True,exist_ok=True)

manifest=[]; class_boxes=Counter()
for i,r in enumerate(records):
    stem=f"{r['source']}_{r['img'].stem}_{i:05d}"
    out_img=OUT/'images'/r['split']/(stem+r['img'].suffix.lower())
    out_lab=OUT/'labels'/r['split']/(stem+'.txt')
    shutil.copy2(r['img'],out_img)
    lines=[]
    if r['lab'].exists():
        for line in r['lab'].read_text().splitlines():
            a=line.split()
            if len(a)>=5:
                # Merge HKUST red/blue/inactive and legacy labels into armor_plate.
                vals=list(map(float,a[1:5]))
                if all(0<=x<=1 for x in vals) and vals[2]>0 and vals[3]>0:
                    lines.append('0 '+' '.join(f'{x:.8f}' for x in vals)); class_boxes[r['split']]+=1
    out_lab.write_text('\n'.join(lines)+('\n' if lines else ''))
    manifest.append([r['split'],r['source'],str(r['img']),str(out_img.relative_to(OUT)),r['group'],int(r['negative']),r['md5'],len(lines)])

(OUT/'data.yaml').write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: armor_plate\n")
with (OUT/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['split','source','original','output','group','negative','md5','boxes']); w.writerows(manifest)

stats=Counter((r['split'],r['source']) for r in records)
print('records',len(records),'negative',sum(r['negative'] for r in records),'boxes',dict(class_boxes))
for k,v in sorted(stats.items()): print(k,v)
