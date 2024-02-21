#include "SAHI.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

// Placeholder function for YOLOv8 inference. Replace with actual YOLOv8 inference code.
std::vector<std::vector<float>> run_yolov8(const cv::Mat& slice) {
    // YOLOv8 inference code should go here
    // This function should return a vector of vectors, where each inner vector has 84 elements (4 for bbox, 80 for class probabilities)
    return {}; // Return empty for placeholder
}

// Function to create mock YOLOv8 output
std::vector<std::vector<float>> create_random_yolov8_output(const cv::Mat& slice, int num_boxes = 20) {
    std::vector<std::vector<float>> mock_output;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> coord_dist(0.0, 1.0); // Normalized coordinates
    std::uniform_real_distribution<> prob_dist(0.0, 1.0);  // Probabilities

    for (int i = 0; i < num_boxes; ++i) {
        std::vector<float> bbox(84); // 4 for bbox coordinates, 80 for class probabilities

        // Generate normalized coordinates for bounding box
        bbox[0] = coord_dist(gen); // x_center
        bbox[1] = coord_dist(gen); // y_center
        bbox[2] = coord_dist(gen) * 0.5; // width (max 20% of image width)
        bbox[3] = coord_dist(gen) * 0.5; // height (max 20% of image height)

        // Randomly assign one class a high probability and others low
        int high_prob_class = std::uniform_int_distribution<>(4, 83)(gen); // Random class index
        for (int j = 4; j < 84; ++j) {
            bbox[j] = (j == high_prob_class) ? prob_dist(gen) * 0.8 + 0.2 : prob_dist(gen) * 0.2;
        }
        mock_output.push_back(bbox);
    }

    return mock_output;
}

// Convert YOLOv8 output to BoundingBox format
std::vector<BoundingBox> process_yolov8_output(const std::vector<std::vector<float>>& yolov8_output, const cv::Mat& slice) {
    std::vector<BoundingBox> bboxes;
    for (const auto& bbox_values : yolov8_output) {
        // Extract and scale bounding box coordinates
        float x_center = bbox_values[0] * slice.cols;
        float y_center = bbox_values[1] * slice.rows;
        float width = bbox_values[2] * slice.cols;
        float height = bbox_values[3] * slice.rows;
        float x = x_center - width / 2;
        float y = y_center - height / 2;

        // Find label with highest probability
        int max_label_idx = std::max_element(bbox_values.begin() + 4, bbox_values.end()) - bbox_values.begin() - 4;
        float max_prob = bbox_values[max_label_idx + 4];

        // Construct BoundingBox
        BoundingBox bbox;
        bbox.rect = cv::Rect_<float>(x, y, width, height);
        bbox.label = max_label_idx;
        bbox.probability = max_prob;

        bboxes.push_back(bbox);
    }
    return bboxes;
}

void theWholeChembangle(const cv::Mat &image, int slice_height, int slice_width,
                        float overlap_height_ratio, float overlap_width_ratio) {
    SAHI sahi(slice_height, slice_width, overlap_height_ratio, overlap_width_ratio);
    std::vector<std::pair<cv::Rect, int>> sliceRegions = sahi.calculateSliceRegions(image.rows, image.cols);

    std::vector<BoundingBox> allBoxes; // For storing all the bounding boxes

    for (const auto& region : sliceRegions) {
        cv::Mat slice = image(region.first);

        // Run YOLOv8 model on the slice
        auto yolov8_output = create_random_yolov8_output(slice);
        std::vector<BoundingBox> sliceDetectedBoxes = process_yolov8_output(yolov8_output, slice);

        cv::Mat sliceWithBoxes = slice.clone();
        for (const auto& box : sliceDetectedBoxes) {
            cv::rectangle(sliceWithBoxes, box.rect, cv::Scalar(0, 255, 0), 2);
            std::string label = std::to_string(box.label) + ": " + std::to_string(box.probability);
            cv::putText(sliceWithBoxes, label, box.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
        }
        cv::imshow("Slice with Bounding Boxes", sliceWithBoxes);
        cv::waitKey(0); // Wait for a key press to continue

        // Store bounding boxes with mapped coordinates
        for (auto& box : sliceDetectedBoxes) {
            // Adjust coordinates to map to the original image
            box.rect.x += region.first.x;
            box.rect.y += region.first.y;
            allBoxes.push_back(box);
        }
    }

    // Apply Non-Maximum Suppression (NMS)
    std::vector<int> indices;
    std::vector<cv::Rect> opencvBoxes;
    std::vector<float> opencvScores;

    for (const auto& box : allBoxes) {
        opencvBoxes.push_back(box.rect);
        opencvScores.push_back(box.probability);
    }

    float nms_threshold = 0.1f; // Adjust as needed
    float score_threshold = 0.1f; // Adjust as needed
    cv::dnn::NMSBoxes(opencvBoxes, opencvScores, score_threshold, nms_threshold, indices);

    // Draw bounding boxes after NMS on the original image
    cv::Mat imageWithBoxes = image.clone();
    for (int idx : indices) {
        const auto& box = allBoxes[idx];
        cv::rectangle(imageWithBoxes, box.rect, cv::Scalar(0, 255, 0), 2);
        std::string label = std::to_string(box.label) + ": " + std::to_string(box.probability);
        cv::putText(imageWithBoxes, label, box.rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }

    cv::imshow("Detected Objects", imageWithBoxes);
    cv::waitKey(0);
}

int main() {
    // Example usage
    cv::Mat image = cv::imread("image.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    int slice_height = 180, slice_width = 320;
    float overlap_height_ratio = 0.2f, overlap_width_ratio = 0.2f;
    theWholeChembangle(image, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio);

    return 0;
}
