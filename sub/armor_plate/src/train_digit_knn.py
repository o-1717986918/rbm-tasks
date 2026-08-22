"""Train a tiny OpenCV KNN digit classifier from labeled crops.
Directory layout: digit_crops/{1,2,3,4,5}/*.png. Produces digit_knn.yml.
"""
import argparse, glob, cv2, numpy as np
from pathlib import Path
def feat(p):
 im=cv2.imread(str(p),0)
 if im is None:return None
 im=cv2.resize(im,(32,20)); _,im=cv2.threshold(im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
 return (im.astype('float32')/255).reshape(1,-1)
ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out',default='digit_knn.yml'); a=ap.parse_args()
X=[]; y=[]
for d in '12345':
 for p in glob.glob(str(Path(a.root)/d/'*')):
  f=feat(p)
  if f is not None:X.append(f[0]); y.append(int(d))
if len(set(y))<5: raise SystemExit('need labeled folders 1,2,3,4,5')
knn=cv2.ml.KNearest_create(); knn.train(np.asarray(X,np.float32),cv2.ml.ROW_SAMPLE,np.asarray(y,np.float32)); knn.save(a.out); print('samples=',len(y),'saved=',a.out)
