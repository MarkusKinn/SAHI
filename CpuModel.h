//
// Created by mktk on 22.02.24.
//

#ifndef CPUMODEL_H
#define CPUMODEL_H

#pragma once
#include "Model.h"
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <opencv2/dnn.hpp>

#include "SAHI.h"

class CpuModel : public InferenceModel {
public:
    CpuModel(const std::string& onnx_model_path);

    std::vector<BoundingBox> run_inference(const cv::Mat& input_image) override;

private:
    cv::Mat preprocess(const cv::Mat& input_image);

    cv::dnn::Net m_net;
    std::string m_output_layer_name;
};



#endif //CPUMODEL_H
