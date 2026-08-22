#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from digit_recognizer import recognize_digit

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--weights',required=True); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True); ap.add_argument('--imgsz',type=int,default=960); ap.add_argument('--conf',type=float,default=.35); ap.add_argument('--digits',action='store_true')
    a=ap.parse_args(); model=YOLO(a.weights); cap=cv2.VideoCapture(a.source)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30; w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outw=min(w,1280); outh=round(h*outw/w); writer=cv2.VideoWriter(a.out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(outw,outh))
    names={0:'armor'}; total=0; frames=0
    while True:
        ok,frame=cap.read()
        if not ok: break
        if (w,h)!=(outw,outh): frame=cv2.resize(frame,(outw,outh))
        result=model.predict(frame,imgsz=a.imgsz,conf=a.conf,device=0,verbose=False)[0]
        for box,cf,cl in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy()):
            x1,y1,x2,y2=map(int,box); cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            tag=f'{names.get(int(cl),"armor")} {cf:.2f}'
            if a.digits:
                d,dc=recognize_digit(frame[y1:y2,x1:x2]); tag += f' ID:{d}'
            cv2.putText(frame,tag,(x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2); total+=1
        writer.write(frame); frames+=1
    cap.release(); writer.release(); print(f'frames={frames} detections={total} output={a.out}')
if __name__=='__main__': main()
