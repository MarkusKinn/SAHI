#include "SAHI.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <random>
#include <functional>

void runYOLOv5(cv::Mat &inputImage) {
    // Load the network
    cv::dnn::Net net = cv::dnn::readNet("yolov5s.onnx");
    std::cout << "Network loaded successfully." << std::endl;


}

int main() {

    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    runYOLOv5(image);

    int sliceHeight = 100, sliceWidth = 100;
    float overlapHeightRatio = 0.2f, overlapWidthRatio = 0.2f;
    SAHI sahi(sliceHeight, sliceWidth, overlapHeightRatio, overlapWidthRatio);

    std::vector<BoundingBoxIndex> allBoxes;

    sahi.slice(image, [&](const cv::Mat& slice, int index) {
        if (slice.empty()) {
            std::cerr << "Error: Slice is empty." << std::endl;
            return;
        }});

    std::vector<cv::Rect> opencvBoxes;
    std::vector<float> opencvScores;
    cv::Mat imageWithBoxes = image.clone();
    for (auto& box : allBoxes) {
        auto region = sahi.calculateSliceRegions(image.rows, image.cols)[box.sliceIndex];
        auto mappedBox = sahi.mapToOriginal(box, region.first);
        opencvBoxes.emplace_back(cv::Rect(mappedBox.x, mappedBox.y, mappedBox.w, mappedBox.h));
        opencvScores.push_back(mappedBox.score);
        cv::rectangle(imageWithBoxes, cv::Rect(mappedBox.x, mappedBox.y, mappedBox.w, mappedBox.h), cv::Scalar(255, 0, 0), 2);
    }

    float iouThreshold = 0.1f;


    // OpenCV NMS Benchmarking
    std::vector<int> indices;
    cv::dnn::NMSBoxes(opencvBoxes, opencvScores, 0.0, iouThreshold, indices);

    // Display the final image after OpenCV NMS
    cv::Mat finalImageOpenCV = image.clone();
    for (int idx : indices) {
        cv::rectangle(finalImageOpenCV, opencvBoxes[idx], cv::Scalar(0, 0, 255), 2);
    }

    return 0;
}

