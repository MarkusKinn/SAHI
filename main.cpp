#include "SAHI.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>

#include "CpuModel.h"


std::vector<cv::Rect> theWholeChembangle(const cv::Mat& image, int slice_height, int slice_width,
                        float overlap_height_ratio, float overlap_width_ratio, CpuModel model) {
    SAHI sahi(slice_height, slice_width, overlap_height_ratio, overlap_width_ratio);
    std::vector<std::pair<cv::Rect, int>> sliceRegions = sahi.calculateSliceRegions(image.rows, image.cols);

    std::vector<BoundingBox> allBoxes; // For storing all the bounding boxes

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
    opencvBoxes.reserve(allBoxes.size());
    opencvScores.reserve(allBoxes.size());

    for (const auto& box : allBoxes) {
        opencvBoxes.push_back(box.rect);
        opencvScores.push_back(box.probability);
    }

    float nms_threshold = 0.1f; // Adjust as needed
    float score_threshold = 0.1f; // Adjust as needed
    cv::dnn::NMSBoxes(opencvBoxes, opencvScores, score_threshold, nms_threshold, indices);

    // Draw bounding boxes after NMS on the original image
    for (int idx : indices) {
        const auto& box = opencvBoxes[idx];
        float probability = opencvScores[idx];
        int label = allBoxes[idx].label;

        cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
        std::string label_text = std::to_string(label) + ": " + std::to_string(probability);
        cv::putText(image, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    cv::imshow("Detected Objects", image);
    cv::waitKey(0);

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

    int slice_height = 180, slice_width = 320;
    float overlap_height_ratio = 0.2f, overlap_width_ratio = 0.2f;
    theWholeChembangle(image, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio, yolov8_model);

    return 0;
}
