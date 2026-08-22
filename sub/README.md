# 视觉识别项目提交包

三个 PyTorch 权重均已导出 ONNX，分别位于各项目的 `weights/` 和 `models/` 目录；导出使用 Ultralytics 8.4.35、ONNX opset 12。

本提交包包含两个并列项目，结构一致：

```text
armor_submission/
├── armor_plate/
│   ├── weights/     # 模型权重
│   ├── demos/       # 推理演示视频
│   ├── dataset/     # 图像、标注、data.yaml
│   ├── src/         # 训练、推理和数字识别代码
│   ├── models/      # 可选 ONNX OCR 模型
│   └── docs/        # 项目说明、来源、划分
└── meme_cat_shield/
    ├── weights/
    ├── demos/
    ├── datasets/
    ├── scripts/
    └── docs/
```

## 项目一：RoboMaster 装甲板

进入 `armor_plate/`，阅读 `docs/README.md`。检测权重、装甲板数据集、两段测试视频、数字识别代码和运行说明均已包含。

## 项目二：妙脆角猫 / 我的刀盾

进入 `meme_cat_shield/`，阅读其 `README.md` 和 `docs/README_Meme_V2.md`。该项目与装甲板项目独立提交，权重、数据、源代码、演示和文档分别存放。
