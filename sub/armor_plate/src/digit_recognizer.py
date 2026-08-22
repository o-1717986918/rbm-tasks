#!/usr/bin/env python3
"""Fast 1-5 digit recognizer for armor crops.

It uses OpenCV only: adaptive thresholding, connected components and a
seven-segment-style projection signature. It is intended as a low-latency
fallback when a digit classifier is unavailable; return '?' when confidence
is low instead of inventing an ID.
"""
import cv2, numpy as np
from pathlib import Path
_REF4=None
for p in [Path(__file__).parent/'digit_templates/4_ref.png', Path('digit_templates/4_ref.png')]:
    if p.exists():
        z=cv2.imread(str(p),0)
        if z is not None: _REF4=z; break

def recognize_digit(crop):
    if crop is None or crop.size == 0: return '?', 0.0
    gray=cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
    gray=cv2.resize(gray,(64,40),interpolation=cv2.INTER_AREA)
    # The digit is normally in the inner plate; suppress the bright border.
    gray=gray[4:-4,8:-8]
    gray=cv2.resize(gray,(64,40),interpolation=cv2.INTER_AREA)
    # Reference-based score for the known blocky RM '4' glyph.
    ref_score=0.0
    if _REF4 is not None:
        rg=cv2.resize(_REF4,(64,40),interpolation=cv2.INTER_AREA)
        rg=cv2.threshold(rg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        a=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        ref_score=max(float(cv2.matchTemplate(a,rg,cv2.TM_CCOEFF_NORMED)[0,0]),
                      float(cv2.matchTemplate(255-a,rg,cv2.TM_CCOEFF_NORMED)[0,0]))
    # Try both polarities: RM plates may have white-on-black or black-on-white glyphs.
    _, inv=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, norm=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw=inv
    # reject crops without a plausible glyph
    n,lab,stats,_=cv2.connectedComponentsWithStats(bw,8)
    comps=[(i,s) for i,s in enumerate(stats[1:],1) if s[4]>12 and s[2]<55 and s[3]<38]
    if not comps: return '?',0.0
    # Keep the largest plausible glyph and remove the plate border.
    i,s=max(comps,key=lambda z:z[1][4]); x,y,wc,hc,area=s
    if area < 0.04*bw.size: # try the opposite polarity for bright digits
        bw=norm; n,lab,stats,_=cv2.connectedComponentsWithStats(bw,8)
        cc=[(j,t) for j,t in enumerate(stats[1:],1) if t[4]>12 and t[2]<55 and t[3]<38]
        if cc: i,s=max(cc,key=lambda z:z[1][4]); x,y,wc,hc,area=s
    roi=bw[max(0,y-2):min(bw.shape[0],y+hc+2),max(0,x-2):min(bw.shape[1],x+wc+2)]
    if roi.size: bw=cv2.resize(roi,(64,40),interpolation=cv2.INTER_NEAREST)
    # normalized occupancy in seven regions (top, middle, bottom, left/right)
    h,w=bw.shape; masks=[(0,0,w//2,h//5),(w//2,0,w,h//5),(0,2*h//5,w//2,3*h//5),(w//2,2*h//5,w,3*h//5),(0,4*h//5,w//2,h),(w//2,4*h//5,w,h),(0,h//5,w,h//2)]
    f=np.array([bw[y1:y2,x1:x2].mean()/255 for x1,y1,x2,y2 in masks])
    templates={'1':np.array([0,0,0,1,0,1,0]),'2':np.array([1,1,1,0,1,0,0]),'3':np.array([1,1,1,1,0,0,0]),'4':np.array([0,0,1,1,0,1,1]),'5':np.array([1,0,1,1,0,1,0])}
    scores={k:1-np.mean(np.abs(f-v)) for k,v in templates.items()}; d=max(scores,key=scores.get); c=float(scores[d])
    # Font-template similarity helps on the blocky white digits used by RM plates.
    best=d; bestc=c
    obs=(bw>0).astype(np.float32)
    for k in '12345':
        t=np.zeros((40,64),np.uint8); cv2.putText(t,k,(15,32),cv2.FONT_HERSHEY_SIMPLEX,1.15,255,2,cv2.LINE_AA)
        t=(t>80).astype(np.float32)
        sim=float((obs*t).sum()/(np.sqrt((obs*obs).sum()*(t*t).sum())+1e-6))
        if sim>bestc: best,bestc=k,sim
    if ref_score>=0.35 and ref_score>bestc: return '4',ref_score
    return (best,bestc) if bestc>=0.42 else ('?',bestc)
