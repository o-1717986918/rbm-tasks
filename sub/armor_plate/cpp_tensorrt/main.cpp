#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nvinfer1;

class Logger final : public ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
  }
};

template <typename T> struct TrtDelete { void operator()(T* value) const { delete value; } };
template <typename T> using TrtPtr = std::unique_ptr<T, TrtDelete<T>>;

static void cudaCheck(cudaError_t code, const char* operation) {
  if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}

static std::vector<char> readBinary(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot open engine: " + path);
  const auto bytes = file.tellg(); file.seekg(0);
  std::vector<char> data(static_cast<size_t>(bytes)); file.read(data.data(), bytes); return data;
}

static size_t elementCount(const Dims& dims) {
  size_t count = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 1) throw std::runtime_error("engine must have static tensor shapes");
    count *= static_cast<size_t>(dims.d[i]);
  }
  return count;
}

struct Detection { cv::Rect box; float confidence; };

class Detector {
 public:
  explicit Detector(const std::string& enginePath) {
    auto bytes = readBinary(enginePath);
    runtime_.reset(createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) throw std::runtime_error("failed to deserialize engine");
    context_.reset(engine_->createExecutionContext());
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char* name = engine_->getIOTensorName(i);
      if (engine_->getTensorDataType(name) != DataType::kFLOAT)
        throw std::runtime_error("sample expects FP32 I/O tensors; TensorRT may still use FP16 internally");
      if (engine_->getTensorIOMode(name) == TensorIOMode::kINPUT) inputName_ = name;
      else outputName_ = name;
    }
    if (inputName_.empty() || outputName_.empty()) throw std::runtime_error("missing engine input/output");
    inputDims_ = engine_->getTensorShape(inputName_.c_str());
    outputDims_ = engine_->getTensorShape(outputName_.c_str());
    inputCount_ = elementCount(inputDims_); outputCount_ = elementCount(outputDims_);
    cudaCheck(cudaMalloc(&inputDevice_, inputCount_ * sizeof(float)), "cudaMalloc input");
    cudaCheck(cudaMalloc(&outputDevice_, outputCount_ * sizeof(float)), "cudaMalloc output");
    cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate");
    if (!context_->setTensorAddress(inputName_.c_str(), inputDevice_) ||
        !context_->setTensorAddress(outputName_.c_str(), outputDevice_))
      throw std::runtime_error("setTensorAddress failed");
    output_.resize(outputCount_);
  }

  ~Detector() {
    if (stream_) cudaStreamDestroy(stream_);
    if (inputDevice_) cudaFree(inputDevice_);
    if (outputDevice_) cudaFree(outputDevice_);
  }

  std::vector<Detection> infer(const cv::Mat& frame, float confThreshold, float iouThreshold) {
    constexpr int size = 640;
    const float scale = std::min(size / static_cast<float>(frame.cols), size / static_cast<float>(frame.rows));
    const int resizedW = std::lround(frame.cols * scale), resizedH = std::lround(frame.rows * scale);
    const int padX = (size - resizedW) / 2, padY = (size - resizedH) / 2;
    cv::Mat resized, letterbox(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::resize(frame, resized, cv::Size(resizedW, resizedH), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(letterbox(cv::Rect(padX, padY, resizedW, resizedH)));
    cv::Mat blob = cv::dnn::blobFromImage(letterbox, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false, CV_32F);
    cudaCheck(cudaMemcpyAsync(inputDevice_, blob.ptr<float>(), inputCount_ * sizeof(float), cudaMemcpyHostToDevice, stream_), "copy input");
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
    cudaCheck(cudaMemcpyAsync(output_.data(), outputDevice_, outputCount_ * sizeof(float), cudaMemcpyDeviceToHost, stream_), "copy output");
    cudaCheck(cudaStreamSynchronize(stream_), "stream sync");

    int channels = 0, anchors = 0; bool channelFirst = true;
    if (outputDims_.nbDims == 3 && outputDims_.d[1] <= 128) {
      channels = outputDims_.d[1]; anchors = outputDims_.d[2]; channelFirst = true;
    } else if (outputDims_.nbDims == 3) {
      anchors = outputDims_.d[1]; channels = outputDims_.d[2]; channelFirst = false;
    } else throw std::runtime_error("unexpected YOLO output rank");
    if (channels < 5) throw std::runtime_error("unexpected YOLO output channels");
    auto value = [&](int anchor, int channel) {
      return channelFirst ? output_[channel * anchors + anchor] : output_[anchor * channels + channel];
    };
    std::vector<cv::Rect> boxes; std::vector<float> scores;
    for (int i = 0; i < anchors; ++i) {
      const float confidence = value(i, 4);
      if (confidence < confThreshold) continue;
      const float cx = value(i, 0), cy = value(i, 1), width = value(i, 2), height = value(i, 3);
      int x = std::lround((cx - width / 2 - padX) / scale);
      int y = std::lround((cy - height / 2 - padY) / scale);
      int w = std::lround(width / scale), h = std::lround(height / scale);
      cv::Rect box(x, y, w, h); box &= cv::Rect(0, 0, frame.cols, frame.rows);
      if (box.area() > 0) { boxes.push_back(box); scores.push_back(confidence); }
    }
    std::vector<int> keep; cv::dnn::NMSBoxes(boxes, scores, confThreshold, iouThreshold, keep);
    std::vector<Detection> result; result.reserve(keep.size());
    for (int index : keep) result.push_back({boxes[index], scores[index]});
    return result;
  }

 private:
  Logger logger_; TrtPtr<IRuntime> runtime_; TrtPtr<ICudaEngine> engine_; TrtPtr<IExecutionContext> context_;
  std::string inputName_, outputName_; Dims inputDims_{}, outputDims_{};
  size_t inputCount_ = 0, outputCount_ = 0; void* inputDevice_ = nullptr; void* outputDevice_ = nullptr;
  cudaStream_t stream_ = nullptr; std::vector<float> output_;
};

class DigitClassifier {
 public:
  explicit DigitClassifier(const std::string& onnx) : net_(cv::dnn::readNetFromONNX(onnx)) {}
  std::pair<std::string, float> infer(const cv::Mat& crop) {
    cv::Mat gray, resized;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY); cv::resize(gray, resized, {32, 32}, 0, 0, cv::INTER_AREA);
    cv::Mat input; resized.convertTo(input, CV_32F, 1.0 / 127.5, -1.0);
    net_.setInput(input.reshape(1, {1, 1, 32, 32})); cv::Mat logits = net_.forward().reshape(1, 1);
    double maxLogit; cv::minMaxLoc(logits, nullptr, &maxLogit);
    std::vector<float> probabilities(6); float total = 0;
    for (int i = 0; i < 6; ++i) total += probabilities[i] = std::exp(logits.at<float>(i) - static_cast<float>(maxLogit));
    int best = static_cast<int>(std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
    float confidence = probabilities[best] / total;
    return {(best > 0 && confidence >= 0.65f) ? std::to_string(best) : "?", confidence};
  }
 private: cv::dnn::Net net_;
};

int main(int argc, char** argv) try {
  if (argc < 5) {
    std::cerr << "usage: " << argv[0] << " detector.engine digit.onnx input.mp4 output.mp4 [conf=0.25] [iou=0.55]\n";
    return 2;
  }
  const float conf = argc > 5 ? std::stof(argv[5]) : 0.25f;
  const float iou = argc > 6 ? std::stof(argv[6]) : 0.55f;
  Detector detector(argv[1]); DigitClassifier digits(argv[2]); cv::VideoCapture capture(argv[3]);
  if (!capture.isOpened()) throw std::runtime_error("cannot open input video");
  const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  const double fps = capture.get(cv::CAP_PROP_FPS);
  cv::VideoWriter writer(argv[4], cv::VideoWriter::fourcc('m','p','4','v'), fps > 0 ? fps : 30, {width, height});
  if (!writer.isOpened()) throw std::runtime_error("cannot create output video");
  cv::Mat frame; size_t frames = 0, detections = 0; double milliseconds = 0;
  while (capture.read(frame)) {
    const auto start = std::chrono::steady_clock::now(); auto found = detector.infer(frame, conf, iou);
    milliseconds += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    for (const auto& detection : found) {
      auto [digit, digitConfidence] = digits.infer(frame(detection.box));
      cv::rectangle(frame, detection.box, {0, 255, 0}, 2);
      const std::string label = "armor " + cv::format("%.2f", detection.confidence) + " ID:" + digit + " " + cv::format("%.2f", digitConfidence);
      cv::putText(frame, label, {detection.box.x, std::max(20, detection.box.y - 5)}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,255,0}, 2);
    }
    detections += found.size(); ++frames; writer.write(frame);
  }
  std::cout << "frames=" << frames << " detections=" << detections << " detector_ms=" << milliseconds / std::max<size_t>(1, frames) << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n'; return 1;
}
