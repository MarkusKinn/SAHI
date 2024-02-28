//
// Created by mktk on 22.02.24.
//

#ifndef YOLOV8_H
#define YOLOV8_H



#pragma once
#include <fstream>
#include <opencv2/core/mat.hpp>

// Utility method for checking if a file exists on disk
inline bool doesFileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

// Config the behavior of the YoloV8 detector.
// Can pass these arguments as command line parameters.
struct YoloV8Config {
    // Calibration data directory. Must be specified when using INT8 precision.
    std::string calibrationDataDirectory;
    // Probability threshold used to filter detected objects
    float probabilityThreshold = 0.25f;
    // Non-maximum suppression threshold
    float nmsThreshold = 0.65f;
    // Max number of detected objects to return
    int topK = 100;
    // Class thresholds (default are COCO classes)
    std::vector<std::string> classNames = {
        "person", "bicycle"
    };
};

class YoloV8 {
public:
    // Builds the onnx model into a TensorRT engine, and loads the engine into memory
    YoloV8(const std::string& onnxModelPath, const YoloV8Config& config);

    // Detect the objects in the image
    cv::Mat detectObjects(const cv::Mat& inputImageBGR);
    cv::Mat detectObjects(const cv::cuda::GpuMat& inputImageBGR);

private:
    // Preprocess the input
    std::vector<std::vector<cv::cuda::GpuMat>> preprocess(const cv::cuda::GpuMat& gpuImg);

    // Postprocess the output
    cv::Mat postprocessDetect(std::vector<float>& featureVector);

    // Used for image preprocessing
    // YoloV8 model expects values between [0.f, 1.f] so we use the following params
    const std::array<float, 3> SUB_VALS {0.f, 0.f, 0.f};
    const std::array<float, 3> DIV_VALS {1.f, 1.f, 1.f};
    const bool NORMALIZE = true;

    float m_ratio = 1;
    float m_imgWidth = 0;
    float m_imgHeight = 0;

    // Filter thresholds
    const float PROBABILITY_THRESHOLD;
    const float NMS_THRESHOLD;
    const int TOP_K;

    // Object classes as strings
    const std::vector<std::string> CLASS_NAMES;
};


#endif //YOLOV8_H
