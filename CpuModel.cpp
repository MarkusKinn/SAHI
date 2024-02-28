#include "CpuModel.h"
#include "yolov8.h"
#include <algorithm>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

CpuModel::CpuModel(const std::string& onnx_model_path) {
    m_net = cv::dnn::readNetFromONNX(onnx_model_path);


    m_output_layer_name = m_net.getUnconnectedOutLayersNames()[0];
    // Debug shapes
    using namespace std;
    using namespace cv;
    using namespace cv::dnn;
    dnn::MatShape input;
    input.push_back(1); input.push_back(3); input.push_back(640); input.push_back(640);
    for(std::string s : m_net.getLayerNames()) {
        cout << s << " " << m_net.getLayerId(s) << endl;;
        vector<dnn::MatShape> in, out;
        m_net.getLayerShapes(input, m_net.getLayerId(s), in, out);
        cout << "IN" << endl;
        for(dnn::MatShape ms : in) {
            for(int i : ms)
                cout << i << " ";
            cout << endl;
        }
        cout << "OUT" << endl;
        for(dnn::MatShape ms : out) {
            for(int i : ms)
                cout << i << " ";
            cout << endl;
        }
    }

}

std::vector<BoundingBox> CpuModel::run_inference(const cv::Mat& input_image) {
    m_image_width = input_image.cols;
    m_image_height = input_image.rows;
    m_scale_x = static_cast<float>(input_image.cols) / InferenceModel::INPUT_WIDTH;
    m_scale_y = static_cast<float>(input_image.rows) / InferenceModel::INPUT_HEIGHT;

    cv::Mat cpu_input = preprocess(input_image);

    m_net.setInput(cpu_input);
    std::cout << cpu_input.size << std::endl;

    std::vector<cv::Mat> cv_output_mat = m_net.forward("output0");

    int out_rows = cv_output_mat[0].size[1];
    int out_cols = cv_output_mat[0].size[2];

    cv::Mat output_mat(out_rows, out_cols, CV_32FC1, (float*)cv_output_mat[0].data);
    output_mat = output_mat.t();

    return process_output(output_mat);
}

cv::Mat CpuModel::preprocess(const cv::Mat& input_image) {
    cv::Mat blob;
    cv::dnn::blobFromImage(
        input_image,
        blob,
        1.0 / 255.0,
        cv::Size(InferenceModel::INPUT_WIDTH, InferenceModel::INPUT_HEIGHT),
        false
    );

    return blob;
}