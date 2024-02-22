#include "Model.h"
#include "yolov8.h"
#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

std::vector<BoundingBox> InferenceModel::process_output(const cv::Mat& output_mat) {
    const int ROWS = output_mat.rows;
    const int NUM_CLASSES = 80; // Specific to Yolov8

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    for (int i = 0; i < ROWS; ++i) {
        auto row_ptr = output_mat.row(i).ptr<float>();

        // First 4 floats in output is the bounding box
        auto bboxes_ptr = row_ptr;
        // The remaining 80 are confidences per class
        auto scores_ptr = row_ptr + 4;
        auto max_score_ptr = std::max_element(scores_ptr, scores_ptr + NUM_CLASSES);

        if (*max_score_ptr > m_probability_threshold) {
            float x = *bboxes_ptr++;
            float y = *bboxes_ptr++;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = std::clamp((x - 0.5f * w) * m_scale_x, 0.f, m_image_width);
            float y0 = std::clamp((y - 0.5f * h) * m_scale_y, 0.f, m_image_height);
            float x1 = std::clamp((x + 0.5f * w) * m_scale_x, 0.f, m_image_width);
            float y1 = std::clamp((y + 0.5f * h) * m_scale_y, 0.f, m_image_height);

            int label = max_score_ptr - scores_ptr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(*max_score_ptr);
        }
    }

    std::vector<BoundingBox> result;

    for (auto& idx : indices) {
        BoundingBox obj{};

        obj.probability = scores[idx];
        obj.label       = labels[idx];
        obj.rect        = bboxes[idx];

        result.push_back(obj);
    }

    return result;
}