# 数据来源、标注与划分

## 装甲检测数据集

最终共 **5,251 张图、8,776 个装甲框**：train 4,275 张/7,111 框，val 440 张/657 框，test 536 张/1,008 框。其中纯背景负样本为 838/50/112 张。

来源：

1. [ansidd/RMPlateDetection](https://github.com/ansidd/RMPlateDetection)：776 张 RoboMaster 对战/实拍帧，原始 YOLO 装甲框。该仓库未见明确数据许可文件，若用于公开商业再分发，应向作者确认授权。
2. [HKUST ENTERPRIZE RM2025 Radar Algorithm](https://github.com/hkustenterprize/RM2025-Radar-Algorithm)：3,504 张公开实拍/比赛装甲检测图；仓库代码为 MIT License。原标签 `inactive/red/blue` 统一映射为单类 `armor_plate`。
3. 同一 HKUST 发布包中的车辆实拍图：在车辆框外采样 1,000 个真实背景区域，生成空 YOLO 标签，覆盖赛场、人员、围挡、直播画面和设备等负类。

合并时对图像内容做 MD5 精确去重，最终从 5,280 个候选得到 5,251 张。划分不是逐图随机：ansidd 按原 batch、HKUST 按相邻编号组、背景按来源帧组，再对组名做固定 SHA-1 哈希，避免连续帧同时落入 train/val/test。`manifest.csv` 记录来源、原始路径、组、MD5、是否负样本和框数。

## 数字分类数据集

最终共 **16,478 张**，类别顺序为 `unknown,1,2,3,4,5`：train 13,333，val 1,604，test 1,541。

来源：

1. [Number-Classifier-for-RoboMaster](https://github.com/baiyeweiguang/Number-Classifier-for-RoboMaster)：RoboMaster 装甲数字/背景灰度裁剪。使用其中目标存在且类别 1～5 的 6,694 张，以及目标不存在的 6,256 张；6～9 不属于本任务，已排除。该仓库未见明确 LICENSE，公开再分发前应向作者确认。
2. HKUST RM2025 Armor Pattern Public Dataset：3,528 张红/蓝装甲 pattern 实拍裁剪。`R1/B1...R5/B5` 映射到数字 1～5，`R0/B0/RS/BS` 映射到 `unknown`。

数字集同样按序列组划分：旧裁剪每连续 50 个编号为一组，HKUST 按文件名前缀为一组，然后固定哈希分为 80%/10%/10%。类 5 样本较少，训练时使用加权采样和几何/模糊/亮度/噪声增强。没有使用水平或垂直翻转。

## 增强和测试隔离

- 检测：`degrees=4, translate=0.12, scale=0.65, perspective=0.0005, mosaic=1.0, mixup=0.05`，最后 10 轮关闭 mosaic。
- 数字：±8°旋转、±2.5 px 平移、0.88～1.12 缩放、轻模糊、亮度/对比度和高斯噪声。
- `VID_20260319_214100.mp4` 与 `VID_20260319_214213.mp4` 完全排除在数据构建和调参之外，只生成最终演示视频。

## 标签格式

- 检测标签：与图片同名 `.txt`，每行 `0 center_x center_y width height`，坐标均归一化到 0～1；负样本标签为空文件。
- 数字标签：目录名和 `manifest.csv` 同时保存类别，`label` 为 0～5，`class_name` 为 `unknown/1/2/3/4/5`。
