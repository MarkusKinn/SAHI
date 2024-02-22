#pragma once
#include "yolov8.h"
#include <vector>

#include "SAHI.h"

class InferenceModel {
public:
    virtual std::vector<BoundingBox> run_inference(const cv::Mat& input_image) = 0;
protected:
    // Logic common to both CPU and GPU
    std::vector<BoundingBox> process_output(const cv::Mat& output_mat);

    static constexpr float INPUT_WIDTH = 640.0;
    static constexpr float INPUT_HEIGHT = 640.0;

    float m_probability_threshold = 0.5f;
    float m_nms_threshold = 0.5f;
    float m_scale_x;
    float m_scale_y;
    float m_image_width;
    float m_image_height;
};
