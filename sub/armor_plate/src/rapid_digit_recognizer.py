import cv2, re
from rapidocr_onnxruntime import RapidOCR

_ocr = RapidOCR()
def recognize_digit(crop):
    if crop is None or crop.size == 0: return '?', 0.0
    crop=cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    result,_ = _ocr(crop)
    best=('?',0.0)
    for item in result or []:
        text=str(item[1]); score=float(item[2])
        m=re.search(r'[1-5]',text)
        if m and score>best[1]: best=(m.group(),score)
    return best
