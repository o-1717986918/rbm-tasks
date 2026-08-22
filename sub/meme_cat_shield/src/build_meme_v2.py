#!/usr/bin/env python3
"""Build a leakage-aware synthetic YOLO dataset for two fixed meme characters."""
from __future__ import annotations
import argparse, csv, hashlib, random, shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
LEGACY_POSITIVE = {
    "miaocuijiao_cat": {"002","006","008","009","010","013","014","015","016","017","018","019","020","021","022","024","025","027","029"},
    "daodun": {"001","004","029","030"},
}

def crop_alpha(im):
    a=np.asarray(im.getchannel("A")); ys,xs=np.where(a>24)
    if not len(xs): raise ValueError("empty mask")
    return im.crop((max(0,xs.min()-2),max(0,ys.min()-2),min(im.width,xs.max()+3),min(im.height,ys.max()+3)))

def white_key(im, threshold=242):
    rgb=np.asarray(im.convert("RGB")); candidate=np.all(rgb>=threshold,axis=2).astype(np.uint8)
    h,w=candidate.shape; flood=candidate.copy(); mask=np.zeros((h+2,w+2),np.uint8)
    for pt in ((0,0),(w-1,0),(0,h-1),(w-1,h-1)):
        if flood[pt[1],pt[0]]: cv2.floodFill(flood,mask,pt,2)
    alpha=(flood!=2).astype(np.uint8)*255
    alpha=cv2.GaussianBlur(cv2.morphologyEx(alpha,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)),(3,3),0)
    return crop_alpha(Image.fromarray(np.dstack([rgb,alpha]),"RGBA"))

def green_key(im):
    rgb=np.asarray(im.convert("RGB")); hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    bg=cv2.inRange(hsv,(35,65,45),(95,255,255))>0
    alpha=(~bg).astype(np.uint8)*255
    alpha=cv2.GaussianBlur(cv2.morphologyEx(alpha,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)),(3,3),0)
    return crop_alpha(Image.fromarray(np.dstack([rgb,alpha]),"RGBA"))

def templates():
    out={0:[],1:[]}
    cats=[ROOT/"sources/user/miaocuijiao_reference.webp"]+[ROOT/f"raw/miaocuijiao_cat/miaocuijiao_cat_{n}.jpg" for n in ("006","013","018","024")]
    for p in cats:
        im=Image.open(p).convert("RGB"); ar=np.asarray(im)
        out[0].append((p.stem,green_key(im) if ar[:,:,1].mean()>ar[:,:,0].mean()*1.25 else white_key(im)))
    dogs=[
        (ROOT/"sources/web/daodun_x2.jpg",None,None),
        (ROOT/"sources/user/daodun_reference.webp",(42,60,248,218),(286,286)),
        (ROOT/"sources/web/daodun_cyberlink.webp",(75,58,710,365),(794,518)),
        (ROOT/"sources/web/daodun_uzzf.png",(115,100,825,430),(860,484)),
    ]
    for p,box,ref in dogs:
        im=Image.open(p).convert("RGB")
        if box:
            sx,sy=im.width/ref[0],im.height/ref[1]
            im=im.crop(tuple(round(v*(sx if i%2==0 else sy)) for i,v in enumerate(box)))
        out[1].append((p.stem,white_key(im)))
    return out

def aug(p,rng):
    a=p.getchannel("A"); rgb=p.convert("RGB")
    rgb=ImageEnhance.Brightness(rgb).enhance(rng.uniform(.7,1.3))
    rgb=ImageEnhance.Contrast(rgb).enhance(rng.uniform(.72,1.28))
    rgb=ImageEnhance.Color(rgb).enhance(rng.uniform(.72,1.25)); rgb.putalpha(a)
    if rng.random()<.28: rgb=rgb.filter(ImageFilter.GaussianBlur(rng.uniform(.4,1.5)))
    if rng.random()<.18 and min(rgb.size)>45:
        f=rng.uniform(.22,.55); s=rgb.resize((max(8,int(rgb.width*f)),max(8,int(rgb.height*f))),Image.Resampling.BILINEAR)
        rgb=s.resize(rgb.size,Image.Resampling.NEAREST)
    return rgb

def background(path,size,rng):
    im=Image.open(path).convert("RGB"); scale=max(size/im.width,size/im.height)*rng.uniform(1,1.35)
    im=im.resize((round(im.width*scale),round(im.height*scale)),Image.Resampling.LANCZOS)
    x=rng.randint(0,max(0,im.width-size)); y=rng.randint(0,max(0,im.height-size))
    return im.crop((x,y,x+size,y+size))

def paste(bg,p,cid,rng):
    size=bg.width; target=rng.randint(max(28,int(size*.055)),int(size*.62)); scale=target/max(p.size)
    p=p.resize((max(4,round(p.width*scale)),max(4,round(p.height*scale))),Image.Resampling.LANCZOS); p=aug(p,rng)
    x=rng.randint(-p.width//4,size-(3*p.width)//4); y=rng.randint(-p.height//5,size-(3*p.height)//4)
    bg.paste(p,(x,y),p); a=np.asarray(p.getchannel("A")); ys,xs=np.where(a>40)
    if not len(xs): return None
    x1,y1=max(0,x+int(xs.min())),max(0,y+int(ys.min())); x2,y2=min(size,x+int(xs.max())+1),min(size,y+int(ys.max())+1)
    if x2-x1<8 or y2-y1<8: return None
    return cid,(x1+x2)/2/size,(y1+y2)/2/size,(x2-x1)/size,(y2-y1)/size

def build(a):
    out=ROOT/a.out
    if out.exists(): shutil.rmtree(out)
    for split in ("train","val","test"):
        (out/"images"/split).mkdir(parents=True); (out/"labels"/split).mkdir(parents=True)
    tm=templates(); coco=sorted((ROOT/"sources/backgrounds/coco128/images/train2017").glob("*.jpg"))
    if len(coco)<100: raise RuntimeError("COCO128 missing")
    groups={"train":coco[:90],"val":coco[90:109],"test":coco[109:]}; nums={"train":a.train,"val":a.val,"test":a.test}; rows=[]
    for si,split in enumerate(("train","val","test")):
        rng=random.Random(a.seed+si*10000)
        for idx in range(nums[split]):
            bgp=rng.choice(groups[split]); bg=background(bgp,a.imgsz,rng); labels=[]; used=[]
            n=0 if rng.random()<.22 else rng.choices([1,2,3],[.70,.23,.07])[0]
            for j in range(n):
                cid=(idx+j)%2; src,p=rng.choice(tm[cid]); lab=paste(bg,p,cid,rng)
                if lab: labels.append(lab); used.append(src)
            stem=f"syn_{split}_{idx:04d}"
            if rng.random()<.30: bg=bg.filter(ImageFilter.GaussianBlur(rng.uniform(.2,.9)))
            bg.save(out/"images"/split/f"{stem}.jpg",quality=rng.randint(45,94),subsampling=rng.choice([0,1,2]))
            (out/"labels"/split/f"{stem}.txt").write_text("\n".join(f"{v[0]} "+" ".join(f"{x:.6f}" for x in v[1:]) for v in labels)+("\n" if labels else ""))
            rows.append((stem,split,"synthetic",bgp.name,";".join(used),len(labels)))
    # Real legacy sources are split once by source ID. Positives use the tighter
    # reviewed labels from meme_det; non-target lookalikes become empty-label
    # hard negatives in every split (never duplicated across splits).
    for folder in ("miaocuijiao_cat","daodun"):
        files=sorted((ROOT/"raw"/folder).glob("*.jpg"))
        for p in files:
            ident=p.stem.rsplit("_",1)[-1]; pos=ident in LEGACY_POSITIVE[folder]
            bucket=int(ident)%10
            split="train" if bucket<7 else ("val" if bucket<9 else "test")
            kind="legacy_real_positive" if pos else "legacy_hard_negative"
            stem=f"{kind}_{folder}_{p.stem}"
            shutil.copy2(p,out/"images"/split/f"{stem}.jpg")
            dest=out/"labels"/split/f"{stem}.txt"
            if pos:
                candidates=list((ROOT/"datasets/meme_det/labels").glob(f"*/{p.stem}.txt"))
                if len(candidates)!=1: raise RuntimeError(f"reviewed label missing for {p.stem}")
                text=candidates[0].read_text().strip()
                # Enforce folder-to-class mapping even if a legacy file drifted.
                cid=0 if folder=="miaocuijiao_cat" else 1
                dest.write_text("\n".join(str(cid)+" "+line.split(maxsplit=1)[1] for line in text.splitlines())+"\n")
                objects=len(text.splitlines())
            else:
                dest.write_text(""); objects=0
            rows.append((stem,split,kind,p.name,"",objects))
            if not pos and split=="train":
                # Oversample source-group-safe lookalikes so ordinary cats/dogs,
                # captions and incomplete weapon variants influence the loss as
                # much as the synthetic positives. These stay in train only.
                rng=random.Random(a.seed+int(ident)+(0 if folder=="miaocuijiao_cat" else 1000))
                src=Image.open(p).convert("RGB")
                for k in range(5):
                    frac=rng.uniform(.72,1.0); cw=max(16,int(src.width*frac)); ch=max(16,int(src.height*frac))
                    x=rng.randint(0,src.width-cw); y=rng.randint(0,src.height-ch)
                    im=src.crop((x,y,x+cw,y+ch)).resize((a.imgsz,a.imgsz),Image.Resampling.LANCZOS)
                    im=ImageEnhance.Brightness(im).enhance(rng.uniform(.72,1.28))
                    if rng.random()<.5: im=im.filter(ImageFilter.GaussianBlur(rng.uniform(.2,1.2)))
                    augstem=f"hardneg_aug_{folder}_{ident}_{k}"
                    im.save(out/"images/train"/f"{augstem}.jpg",quality=rng.randint(42,88))
                    (out/"labels/train"/f"{augstem}.txt").write_text("")
                    rows.append((augstem,"train","hard_negative_augmentation",p.name,"",0))
    (out/"data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: miaocuijiao_cat\n  1: sword_shield_dog\n")
    with (out/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["image","split","kind","background_or_source","template_sources","objects"]); w.writerows(rows)
    sums=[]
    for p in sorted(out.rglob("*")):
        if p.is_file(): sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}")
    (out/"SHA256SUMS.txt").write_text("\n".join(sums)+"\n")
    for s in ("train","val","test"): print(s,len(list((out/"images"/s).glob("*"))))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="datasets/meme_v2"); ap.add_argument("--imgsz",type=int,default=640)
    ap.add_argument("--train",type=int,default=720); ap.add_argument("--val",type=int,default=144); ap.add_argument("--test",type=int,default=144); ap.add_argument("--seed",type=int,default=260822)
    build(ap.parse_args())
