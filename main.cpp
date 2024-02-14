#include "SAHI.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int yeet = 0;

std::vector<BoundingBox> runYOLOv8(const cv::Mat &inputImage, cv::dnn::Net &net) {
    // Preprocess the image as required by YOLOv8
    std::cout << "Image inferred successfully: " << yeet << std::endl;
    ++yeet;
    cv::Mat blob;
    cv::dnn::blobFromImage(inputImage, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(0, 0, 0), true, false);

    // Set the input to the network
    net.setInput(blob);

    // Forward pass to get the output
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Process outputs to detect bounding boxes
    std::vector<BoundingBox> detectedBoxes;
    for (const auto& output : outputs) {
        // The shape of the output could be something like [number_of_detections, 6]
        // where each detection is composed of [x, y, width, height, confidence, class_id]
        const float* data = reinterpret_cast<float*>(output.data);
        for (int j = 0; j < output.rows; ++j, data += output.cols) {
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            float score = data[4];
            // Optionally, you can also use class_id = data[5], if needed

            // Apply a confidence threshold
            if (score > 0.5) {
                // Convert box from [center_x, center_y, width, height] to [top_left_x, top_left_y, width, height]
                float x0 = x - w / 2;
                float y0 = y - h / 2;

                detectedBoxes.emplace_back(x0, y0, w, h, score);
            }
        }
    }

    return detectedBoxes;
}


void theWholeChembangle(const cv::Mat &image, cv::dnn::Net &yolo, int slice_height, int slice_width,
        float overlap_height_ratio, float overlap_width_ratio) {

    SAHI sahi(slice_height, slice_width, overlap_height_ratio, overlap_width_ratio);
    std::vector<std::pair<cv::Rect, int>> sliceRegions = sahi.calculateSliceRegions(image.rows, image.cols);

    std::vector<BoundingBox> allBoxes; // For storing all the bounding boxes
    std::vector<float> opencvScores; // For storing scores of bounding boxes

    for (const auto& region : sliceRegions) {
        cv::Mat slice = image(region.first);

        cv::imshow("Slice", slice);
        cv::waitKey(0);

        std::vector<BoundingBox> sliceDetectedBoxes = runYOLOv8(slice, yolo);

        for (const auto& box : sliceDetectedBoxes) {
            // Map the bounding box to the original image's coordinate system
            BoundingBox mappedBox = SAHI::mapToOriginal(box, region.first);
            allBoxes.push_back(mappedBox);
            opencvScores.push_back(mappedBox.score);
        }
    }

    std::vector<cv::Rect> opencvBoxes;
    for (const auto& box : allBoxes) {
        opencvBoxes.emplace_back(cv::Point(box.x, box.y), cv::Size(box.w, box.h));
    }

    // Non-Maximum Suppression
    std::vector<int> indices;
    float score_threshold = 0.5; // Adjust this threshold based on your requirements
    float nms_threshold = 0.4; // Adjust this threshold based on your requirements
    cv::dnn::NMSBoxes(opencvBoxes, opencvScores, score_threshold, nms_threshold, indices);

    cv::Mat finalResult = image.clone();
    for (int idx : indices) {
        const auto& box = opencvBoxes[idx];
        cv::rectangle(finalResult, box, cv::Scalar(0, 255, 0), 10);
    }
    cv::imshow("Final Result", finalResult);
    cv::waitKey(0);
}

int main() {
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX("yolov5s.onnx");
    if (net.empty()) {
        std::cerr << "Error: Could not load YOLOv5 model." << std::endl;
        return -1;
    }

    int slice_height = 180, slice_width = 320;
    float overlap_height_ratio = 0.0f, overlap_width_ratio = 0.0f;
    theWholeChembangle(image, net, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio);

    return 0;
}
