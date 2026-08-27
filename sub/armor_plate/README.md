# RoboMaster 装甲板检测与数字 ID 识别

本目录是可复现提交包：先用 YOLO11n 检测装甲板，再用 29,558 参数的 TinyDigitNet 识别 `1～5`；数字置信度不足时输出 `?`，并用连续帧投票降低跳变。检测器训练和推理均固定为 **640×640**，便于 ONNX/TensorRT 静态图部署。

## 提交内容

| 目录/文件 | 内容 |
|---|---|
| `weights/armor_yolo11n_640_best.pt` | 完整 PyTorch 检测权重 |
| `weights/armor_yolo11n_640.onnx` | 静态 `1×3×640×640` 检测模型 |
| `weights/armor_digit_tiny_best.pt` | 数字分类器训练检查点 |
| `weights/armor_digit_tiny.onnx` | 静态 `1×1×32×32` 数字分类模型 |
| `demo/*.mp4` | 两段用户测试视频的新模型推理结果 |
| `dataset/armor_detection/` | 5,251 张检测图及 YOLO 标签 |
| `dataset/armor_digits/` | 16,478 张数字/负类图及划分清单 |
| `src/` | 数据构建、训练、评估和 Python 视频推理 |
| `cpp_tensorrt/` | TensorRT Engine 构建脚本和 C++/CUDA 推理程序 |
| `metrics/` | 检测器、数字分类器及分距离评估结果 |

## 模型质量

检测器在完全独立的 test 划分上以 640 输入评估：536 张、1,008 个框，**Precision 92.35%、Recall 86.51%、mAP50 90.95%、mAP50-95 60.01%**。RTX 5060 上批量评估的平均时间为预处理 0.98 ms、网络推理 2.21 ms、后处理 0.82 ms/图。数字模型 test 集共 1,541 张：准确率 **97.01%**，类别 `unknown,1,2,3,4,5` 的召回率分别为 **99.42%、95.74%、97.48%、94.57%、92.27%、96.30%**。数字 ONNX 经 OpenCV DNN 复测为 **97.08%**，与 PyTorch 一致。

距离适应性不再用“训练 640、推理 960”的隐式放大补救。以 `conf=0.25, IoU≥0.5` 统计，装甲原始宽度 `<24 px / 24～64 px / ≥64 px` 的召回率分别为 **72.44% / 96.84% / 85.39%**；112 张纯背景中 4 张出现检测框，背景图误报率 **3.57%**。远距离目标本质上像素不足，若部署验收仍要求更远距离，可在相同 640 输入下做 ROI 跟踪二次放大，而不要直接更换网络输入尺寸。

## 输入、输出与后处理

### 装甲检测

- 输入：BGR 图像；保持比例 letterbox 到 `640×640`，填充值 114；BGR→RGB；除以 255；NCHW。
- ONNX 输入：`images`, `float32[1,3,640,640]`。
- ONNX 原始输出：YOLO11 检测张量，通常为 `float32[1,5,8400]`；每列是 `cx,cy,w,h,class_score`。
- 后处理：建议 `confidence=0.25`、`NMS IoU=0.55`；现场若人衣服/文字误报偏多，优先将 confidence 调到 `0.35～0.45`。
- 最终框：`x1,y1,x2,y2,confidence,class_id`，本模型只有 `class_id=0 (armor_plate)`。

### 数字识别

- 输入：检测框内完整装甲图，灰度化并直接缩放到 `32×32`，归一化为 `(pixel/255-0.5)/0.5`。
- ONNX 输入/输出：`images float32[1,1,32,32]` → `logits float32[1,6]`。
- 类别顺序：`unknown,1,2,3,4,5`。
- 后处理：softmax；最大类不是 `unknown` 且概率≥0.65 才输出数字，否则输出 `?`；演示脚本对最近 7 帧概率取均值。
- 敌方判断：脚本另用 HSV 统计红/蓝灯条像素；正式机器人应由己方颜色配置决定敌方，不要仅依赖单帧颜色。

## 快速运行

```bash
pip install -r requirements.txt
python src/infer_video.py \
  --weights weights/armor_yolo11n_640_best.pt \
  --digit-model weights/armor_digit_tiny.onnx \
  --source input.mp4 --out result.mp4 \
  --imgsz 640 --conf 0.25 --iou 0.55 --device 0
```

仅使用 CPU/ONNX 时，可按 `cpp_tensorrt/main.cpp` 的预处理和解析方式改为 OpenCV DNN；本机 ONNX Runtime CPU 单图网络推理约 59.1 ms，精度为 mAP50 90.96%、mAP50-95 59.08%。实际 NVIDIA 开发板优先使用 TensorRT FP16。

## 复现训练

环境：WSL Ubuntu 22.04、Python 3.9、Ultralytics 8.4.35、PyTorch 2.8.0+cu128；训练硬件为 NVIDIA RTX 5060 Laptop 8GB。检测训练峰值显存约 4.6GB，因此本机足够，不需要租用算力服务。WSL 内存较小时不要启用 `cache=ram`。

```bash
bash src/train_detector.sh dataset/armor_detection/data.yaml yolo11n.pt

python src/train_digit_classifier.py \
  --number-root external/Number-Classifier/dataset/extracted/armors \
  --hkust-root external/hkust-pattern \
  --output runs/digit_classifier --epochs 30 --batch 256
```

检测训练使用 AdamW、最多 60 epochs、patience 12、batch 32、AMP、余弦学习率，并针对小目标使用平移/缩放/轻透视、mosaic 和少量 mixup。数字分类训练不做水平翻转，因为翻转会破坏数字语义。

## TensorRT/C++

见 [`cpp_tensorrt/README.md`](cpp_tensorrt/README.md)。提交的是跨设备可迁移的 ONNX；`.engine` 与 GPU 架构、TensorRT 和 CUDA 版本绑定，应在目标开发板运行 `build_engine.sh` 重新生成。当前 WSL 未安装 TensorRT SDK，因此没有把一份不可移植的本机 Engine 冒充开发板权重。

## 数据来源和限制

完整来源、类别映射、许可提示、去重和划分方式见 [`DATASET_SOURCES.md`](DATASET_SOURCES.md)。用户提供的两段测试视频只用于最终演示，**没有进入训练集、验证集或测试集**。

## 验收建议

1. 先用 `confidence=0.35` 跑开发板视频，记录端到端 P50/P95 延迟和实际丢帧率。
2. 同时查看远/中/近目标召回、纯背景误报和 ID 混淆，不只看总体 mAP。
3. 若相机、曝光或装甲字体与公开数据差异大，采集现场序列并以“整段序列”为单位重新划分，避免相邻帧泄漏。
