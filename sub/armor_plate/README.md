# RoboMaster 装甲板检测


## 训练

```bash
conda activate yolo_env
yolo detect train model=/home/win98/my_projects/yoloproj/rbmproject/yolo11n.pt data=/home/win98/my_projects/yoloproj/armorproj/datasets/armor/data.yaml imgsz=640 epochs=80 patience=15 batch=16 device=0 workers=8 cache=ram amp=True optimizer=AdamW lr0=0.001 cos_lr=True close_mosaic=10 project=/home/win98/my_projects/yoloproj/armorproj/runs name=armor_yolo11n
```

## 视频推理

```bash
python scripts/infer_video.py --weights runs/armor_yolo11n/weights/best.pt --source videos/VID_20260319_214213.mp4 --out outputs/armor_demo_214213.mp4 --imgsz 960 --conf 0.35
```

输入为竖屏 1080×2368 视频；脚本把输出宽度限制为 1280，保持比例。输出检测框格式为 `x1,y1,x2,y2,confidence,class`，当前模型类别只有 `armor_plate`。

## 数字识别说明

公开数据的标签只有装甲板框，没有可靠的 1～5 数字标注。因此数字模块提供可部署的 OpenCV 方案，并提供 `train_digit_knn.py`：补充 `digit_crops/1..5` 后即可训练 `digit_knn.yml`，不把未标注数据伪装成数字标签。

## 本次补充的数字识别方案

`digit_recognizer.py` 提供纯 OpenCV 的低延迟 1～5 识别器：对装甲板框进行灰度化、Otsu 二值化、区域投影，与数字模板签名匹配；置信度不足时输出 `?`，避免误报。视频推理可加 `--digits` 开启：

```bash
python scripts/infer_video.py --weights armor_yolo11n_best.pt --source input.mp4 --out result.mp4 --digits
```

该方案不需要额外模型，适合开发板低延迟部署；若现场字体、倾斜或反光差异较大，应采集每个 ID 至少 100 张清晰裁剪图，按 `1/2/3/4/5` 建目录训练一个小型 MobileNet/ResNet 分类器，并替换此模块。当前公开数据的数字标签不足，未将伪标签冒充人工标注。

训练数字 KNN：`python train_digit_knn.py --root digit_crops --out digit_knn.yml`。装甲板重训脚本为 `train_armor.sh`。

## 提交内容清单

- `armor_yolo11n_best.pt`：训练权重；`armor_dataset/`：含 `images/train|val` 与 YOLO 标注。
- `armor_demo_*.mp4`：两段测试视频推理结果。
- `infer_video.py`、`digit_recognizer.py`：推理和数字识别源代码。
- `SOURCES.md`：数据来源、许可核验提示和划分说明。
- `meme_cat_shield/`：之前的妙脆角猫/刀盾项目完整材料。

## 运行环境与输出

已在 WSL Ubuntu-22.04、Python/conda `yolo_env`、Ultralytics 8.4.35、Torch 2.8 CUDA、RTX 5060 Laptop 上验证。输入为普通 BGR 视频；输出为 MP4，检测框格式为 `x1,y1,x2,y2,confidence,class`，数字模式额外显示 `ID:1~5` 或 `ID:?`。建议装甲板阈值 `conf=0.25~0.35`，数字结果必须结合连续帧投票后再作敌方 ID 判断。

## 部署

```bash
yolo export model=runs/armor_yolo11n/weights/best.pt format=onnx imgsz=640 simplify=True opset=12
```

部署板上优先使用 640 输入、FP16/ONNX；960 用于远处小装甲板的精度测试。`conf=0.35` 是演示起点，实际部署应在开发板上按漏检/误检和延迟重新标定。

# 数据来源

- `datasets/armor/images` and `labels`: [ansidd/RMPlateDetection](https://github.com/ansidd/RMPlateDetection), downloaded 2026-08-23. Its README describes YOLO-converted RoboMaster gameplay frames with armor annotations. Verify repository licensing before redistribution.
- `videos/VID_20260319_214100.mp4`, `videos/VID_20260319_214213.mp4`: user-provided videos, excluded from training and used only for final demonstration.

The public data are kept in original train/val folders. No frames from the two user videos are included in either split.