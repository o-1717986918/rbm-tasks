#!/usr/bin/env python3
"""Evaluate the final detector, including hard-negative false positives."""
import argparse, csv, json
from pathlib import Path
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--weights",required=True); ap.add_argument("--conf",type=float,default=.35); ap.add_argument("--imgsz",type=int,default=640)
    a=ap.parse_args(); model=YOLO(a.weights); out=ROOT/"reports"/"meme_v2"; out.mkdir(parents=True,exist_ok=True)
    metrics=model.val(data=str(ROOT/"datasets/meme_v2/data.yaml"),split="test",imgsz=a.imgsz,batch=16,device=0,plots=True,project=str(out),name="test",exist_ok=True)
    manifest={r["image"]:r for r in csv.DictReader((ROOT/"datasets/meme_v2/manifest.csv").open(encoding="utf-8"))}
    hard=[]
    for stem,r in manifest.items():
        if r["kind"]=="legacy_hard_negative": hard.append((r["split"],ROOT/"datasets/meme_v2/images"/r["split"]/(stem+".jpg")))
    fp=[]
    for (split,path),result in zip(hard,model.predict([p for _,p in hard],conf=a.conf,imgsz=a.imgsz,device=0,verbose=False)):
        if len(result.boxes): fp.append({"image":path.name,"split":split,"detections":len(result.boxes),"max_conf":float(result.boxes.conf.max())})
    refs=[ROOT/"sources/user/miaocuijiao_reference.webp",ROOT/"sources/user/daodun_reference.webp"]
    ref_results=[]
    for path,result in zip(refs,model.predict(refs,conf=a.conf,imgsz=a.imgsz,device=0,save=True,project=str(out),name="references",exist_ok=True,verbose=False)):
        ref_results.append({"image":path.name,"detections":[{"class":int(c),"confidence":float(cf),"xyxy":[round(float(v),2) for v in box]} for c,cf,box in zip(result.boxes.cls,result.boxes.conf,result.boxes.xyxy)]})
    unseen=[x for x in hard if x[0]=="test"]; unseen_fp=[x for x in fp if x["split"]=="test"]
    report={"weights":str(Path(a.weights).resolve()),"confidence_threshold":a.conf,"test":{"precision":float(metrics.box.mp),"recall":float(metrics.box.mr),"mAP50":float(metrics.box.map50),"mAP50_95":float(metrics.box.map)},"hard_negatives":{"all_images":len(hard),"all_false_positive_images":len(fp),"unseen_images":len(unseen),"unseen_false_positive_images":len(unseen_fp),"details":fp},"references":ref_results}
    (out/"evaluation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
