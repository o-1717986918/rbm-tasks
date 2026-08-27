# TensorRT + C++ 部署

检测器使用静态 `1×3×640×640` ONNX 构建 FP16 TensorRT Engine；数字分类器是约 7 万参数的 `1×1×32×32` ONNX，由 OpenCV DNN 执行。TensorRT Engine 与 GPU 架构、TensorRT/CUDA 版本相关，因此提交 ONNX 和构建脚本，必须在目标开发板上重新生成 Engine。

```bash
chmod +x build_engine.sh
./build_engine.sh ../weights/armor_yolo11n_640.onnx armor_fp16.engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/rm_armor armor_fp16.engine ../weights/armor_digit_tiny.onnx input.mp4 output.mp4 0.25 0.55
```

代码执行 YOLO letterbox、RGB/255、TensorRT 推理、置信度筛选、NMS、坐标反变换，以及数字分类。正式上板应增加 CUDA 预处理、页锁定内存、异步双缓冲和连续帧 ID 投票，并在目标板实测 P50/P95 延迟、吞吐和丢帧率。
