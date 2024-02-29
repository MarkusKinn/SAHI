#include "SAHI.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>

#include "CpuModel.h"

// cutout_img
// uint32 centerx/y
// float32 confidence
// bool is_emergent

float NMS_THRESHOLD = 0.1f;
float SCORE_THRESHOLD = 0.1f;

std::vector<cv::Rect> theWholeChembangle(const cv::Mat& image, int slice_height, int slice_width,
                        float overlap_height_ratio, float overlap_width_ratio, CpuModel model) {
    SAHI sahi(slice_height, slice_width, overlap_height_ratio, overlap_width_ratio);
    std::vector<std::pair<cv::Rect, int>> sliceRegions = sahi.calculateSliceRegions(image.rows, image.cols);

    std::vector<BoundingBox> allBoxes;

    for (const auto& region : sliceRegions) {

        cv::Mat slice = image(region.first);

        // Run YOLOv8 model on the slice
        std::vector<BoundingBox> sliceDetectedBoxes = model.run_inference(slice);

        // Store bounding boxes with mapped coordinates
        for (auto& box : sliceDetectedBoxes) {
            box = SAHI::mapToOriginal(box, region.first);
            allBoxes.push_back(box);
        }
    }

    std::vector<int> indices;
    std::vector<cv::Rect> opencvBoxes;
    std::vector<float> opencvScores;

    for (const auto& box : allBoxes) {
        opencvBoxes.push_back(box.rect);
        opencvScores.push_back(box.probability);
    }

    cv::dnn::NMSBoxes(opencvBoxes, opencvScores, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    return opencvBoxes;
}

int main() {
    // Example usage
    std::string model_path = "plz_work.onnx";
    CpuModel yolov8_model(model_path);

    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    int slice_height = 360, slice_width = 640;
    float overlap_height_ratio = 0.0f, overlap_width_ratio = 0.0f;
    theWholeChembangle(image, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, yolov8_model);

    return 0;
}
