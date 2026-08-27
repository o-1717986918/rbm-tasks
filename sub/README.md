# 视觉识别项目提交包

本提交包包含两个并列、互相独立的项目：

```text
sub/
├── armor_plate/       # RoboMaster 装甲板检测 + 数字 1～5 识别（重点）
│   ├── weights/       # PT 与 ONNX
│   ├── demo/          # 两段最终演示视频
│   ├── dataset/       # 检测集与数字分类集
│   ├── src/           # 构建、训练、评估、推理源码
│   ├── cpp_tensorrt/  # TensorRT Engine 构建和 C++ 推理
│   ├── metrics/       # 独立测试与距离分桶结果
│   └── README.md
└── meme_cat_shield/   # 妙脆角猫 / 我的刀盾识别
    ├── weights/
    ├── demos/
    ├── datasets/
    ├── scripts/
    └── README.md
```

装甲板项目训练和推理统一为 640×640，并加入实拍数据、背景负类、分组划分、轻量数字 CNN、ONNX、TensorRT/C++ 部署材料。先阅读 [`armor_plate/README.md`](armor_plate/README.md)。

妙脆角猫/刀盾项目保持独立提交，阅读 [`meme_cat_shield/README.md`](meme_cat_shield/README.md)。
