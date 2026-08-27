# RoboMaster 装甲板识别：数据、训练与 TensorRT/C++ 部署计划

更新日期：2026-08-27

> 完成状态：实拍数据合并、1,000 张背景负类、分组划分、640×640 统一训练、轻量数字 CNN、独立测试、PT/ONNX、两段演示视频及 TensorRT/C++ 源码均已完成。目标板专属 TensorRT Engine 仍应在目标设备上由提交的 ONNX 构建。

## 1. 当前问题

- 当前数据集仅 776 张实拍图片，来源单一，缺少足够负样本。
- 当前仅有 `armor_plate` 单类，机器人背面、熄灭灯条、LED、反光和场地文字容易造成误检。
- 训练尺寸为 640，Python 演示默认使用过 960，现有静态 ONNX 为 640，质量与速度数据不能直接对齐。
- 远距离装甲板像素过少，近、中、远距离的召回率和定位精度差异明显。
- OpenCV 数字模板识别仅适合作为备用规则，不能作为可靠的敌方 ID 分类器。

## 2. 目标架构

```text
相机画面
  -> YOLO11n 装甲板检测
  -> 机器人区域、颜色与灯条几何辅助校验
  -> 装甲板透视校正
  -> MobileNetV2/V3-Small 数字 1~5/unknown 分类
  -> 连续帧跟踪与 ID 投票
```

OpenCV负责颜色、灯条几何、透视变换、ROI 和跟踪；数字本身使用轻量分类器。

## 3. 数据扩充

优先审计并接入以下实拍来源：

1. HKUST ENTERPRIZE 2025：约 3504 张装甲板检测数据、3528 张图案分类数据及真实第一视角视频，MIT License。
   - https://github.com/hkustenterprize/RM2025-Radar-Algorithm
2. XJTLU RoboMaster 2023：约 20500 张装甲板/视频帧数据，含装甲板四角点和数字类别，Apache-2.0。
   - https://github.com/zRzRzRzRzRzRzR/YOLO-of-RoboMaster-Keypoints-Detection-2023
3. Roboflow RoboMaster Armor Plate：约 23282 张红/蓝装甲板图片，CC BY 4.0；下载前检查版本、重复率和标注质量。
   - https://universe.roboflow.com/sam-fplas/robomaster-armor-plate
4. PolySTAR RoboMaster 2022 CV / DJI ROCO：真实比赛画面，包含红、蓝、灰装甲板及机器人框。
   - https://github.com/PolySTAR-mtl/robomaster-2022-cv
5. 保留现有 ansidd/RMPlateDetection 数据作为补充来源。

目标不是无筛选堆叠，而是形成约 8000~15000 张高质量正样本、2000~4000 张负样本，以及每个数字至少 1000 张裁剪图。

### 3.1 负样本

检测任务不增加“背景”类别，而是使用空标签图片。负样本应包含空场地、机器人背面/顶部、熄灭装甲板、单灯条、LED、反光、屏幕、围栏、数字、队服、人员和当前模型高置信度误检帧。建议负样本占训练图片 20%~30%，每轮训练后继续 hard-negative mining。

### 3.2 数据划分

- 按视频、比赛、队伍、拍摄时间和场景分组划分，禁止随机拆分相邻帧。
- 两段指定测试视频始终排除在训练集外。
- 按部署尺寸下的装甲板像素宽高将样本分为远/中/近三个桶，分别报告 AP 和 Recall。
- 对图片和视频帧做感知哈希/MD5 去重，派生图不得跨集合。

### 3.3 增强

使用运动模糊、失焦、视频压缩、LED 过曝/光晕、欠曝、Gamma、色温、轻微透视、遮挡和小目标缩放增强。避免上下翻转和不符合机器人运动范围的强旋转；训练最后约 10 个 epoch 关闭 Mosaic。

## 4. 输入尺寸统一

先比较固定 512、640、768 三种尺寸，然后选择一个正式部署尺寸。正式版本必须满足：

```text
训练尺寸 = 验证尺寸 = ONNX尺寸 = TensorRT尺寸 = C++ letterbox尺寸
```

默认以 640 为基线。960 仅作离线精度实验，不再作为默认演示尺寸。为了让 TensorRT 选择更优 tactic，优先为 512/640/768 分别建立固定 `MIN=OPT=MAX` engine，而不是一个跨度很大的动态 engine。

## 5. 模型实验顺序

1. Baseline：YOLO11n、640、单类装甲板、新数据和负样本。
2. 颜色模型：`red_armor/blue_armor/inactive_armor`。
3. 辅助判断：机器人 ROI、灯条宽高比/角度/间距/颜色校验及连续帧跟踪。
4. 小目标：先增加远距离数据，再比较 768；仍不足时才尝试 P2 检测头或四角点模型。
5. 数字：训练 `unknown/1/2/3/4/5` 的 MobileNetV2 或 MobileNetV3-Small，输入 48 或 64。

每次只改变一个变量，保存相同 test 集上的精度、速度和误检对比。

## 6. TensorRT 与 C++

- 静态 ONNX，batch=1，opset 12。
- 在最终部署 GPU、相同 CUDA/TensorRT 版本上构建 engine，不跨硬件复制 `.engine`。
- 依次比较 FP32、FP16 和 INT8；INT8 使用 500~1000 张覆盖远近、红蓝、明暗的校准图片。
- 若 INT8 的 mAP50-95 下降超过约 1~2 个百分点，保留 FP16。
- 使用 `trtexec` 测量预热后延迟、吞吐量和逐层耗时。
- C++使用预分配 CPU/GPU 缓冲、pinned memory、CUDA stream、`enqueueV3`，避免逐帧分配内存。
- 固定尺寸稳定后测试 CUDA Graph；NMS 尽量放 GPU。
- 相机、推理、显示/串口使用解耦线程，队列只保留最新帧，避免延迟累计。

## 7. 验收指标

- 全集及近/中/远距离 Precision、Recall、mAP50、mAP50-95。
- 红/蓝/熄灭装甲板指标和负样本每分钟误检数。
- 数字 1~5/unknown 混淆矩阵和连续帧 ID 准确率。
- `.pt`、ONNX、TensorRT FP32/FP16/INT8 精度对比。
- C++预处理、H2D、推理、NMS、数字分类和端到端 p50/p95 延迟。
- FPS、丢帧率、GPU/CPU占用、显存、功耗及连续运行稳定性。
- 两段指定视频的最终推理演示。

## 8. 实施顺序

1. 下载和审计数据及许可证。
2. 统一类别、转换标注、去重并添加负样本。
3. 建立按场景隔离和距离分桶的固定 train/val/test。
4. 训练 640 新基线并完成误检挖掘。
5. 比较 512/640/768，确定部署尺寸。
6. 训练数字轻量分类器。
7. 导出静态 ONNX，并在目标硬件构建 TensorRT FP16。
8. 实现 C++完整流水线。
9. 测试 INT8、CUDA Graph及流水线优化。
10. 重新生成权重、演示、指标和最终文档。

## 9. 本机训练算力判断

本机为 NVIDIA GeForce RTX 5060 Laptop GPU，显存 8151 MiB。它足够完成 YOLO11n/YOLO11s 和轻量数字分类器的主要训练、数据清洗、消融实验及 ONNX/TensorRT 开发。

推荐设置：

- YOLO11n，640：batch 8~16，AMP/FP16，显存压力较小。
- YOLO11n，768：batch 4~8，必要时梯度累积。
- YOLO11s，640：batch 4~8，根据实测自动降 batch。
- MobileNet 数字分类：batch 64~256，8GB 显存足够。
- 数据集增大不会直接增加单步显存，只增加 epoch 时间和磁盘/缓存需求。

当前不必立即购买算力服务。只有在并行搜索大量超参数、训练 YOLO11m 以上模型、使用 960/1280 大输入、数万图多轮快速迭代，或希望把数天实验压缩到数小时的情况下，才建议临时租用 24GB 以上 GPU。最终 TensorRT engine 仍必须在目标部署设备上构建和验证。
