#include <opencv2/opencv.hpp>
#include <random>
#include "NMS.h"

// Function to generate random bounding boxes
std::vector<BoundingBox> generateOverlappingBoxes() {
    std::vector<BoundingBox> boxes;

    // High score box - likely to be kept
    boxes.emplace_back(50, 50, 100, 100, 0.9, 0); // x, y, w, h, score, sliceIndex

    // Overlapping boxes with lower scores - likely to be removed
    boxes.emplace_back(60, 60, 100, 100, 0.7, 0); // Overlaps with the first box
    boxes.emplace_back(40, 40, 100, 100, 0.6, 2); // Overlaps with the first box
    boxes.emplace_back(55, 55, 100, 100, 0.5, 3); // Overlaps with the first box

    // Another high score box, no overlap with the first group
    boxes.emplace_back(200, 200, 80, 80, 0.95, 4);

    // Overlapping boxes with lower scores - likely to be removed
    boxes.emplace_back(210, 210, 80, 80, 0.65, 5); // Overlaps with the second high score box
    boxes.emplace_back(190, 190, 80, 80, 0.55, 6); // Overlaps with the second high score box

    return boxes;
}



// Function to draw bounding boxes on an image
void drawBoxes(cv::Mat& img, const std::vector<BoundingBox>& boxes, const cv::Scalar& color) {
    for (const auto& box : boxes) {
        cv::rectangle(img, cv::Point(box.x, box.y), cv::Point(box.x + box.w, box.y + box.h), color, 2);
    }
}

int main() {
    // Load an image
    cv::Mat img = cv::imread("test_640.jpg");

    if (img.empty()) {
        std::cerr << "Error: Image not found." << std::endl;
        return -1;
    }

    // Generate random bounding boxes
    std::vector<BoundingBox> boxes = generateOverlappingBoxes();

    // Draw boxes on the image
    cv::Mat imgWithBoxes = img.clone();
    drawBoxes(imgWithBoxes, boxes, cv::Scalar(0, 255, 0)); // Green color for initial boxes

    // Create an NMS object and apply NMS
    NMS nms;
    float iouThreshold = 0.5;
    std::vector<BoundingBox> filteredBoxes = nms.nonMaximumSupression(boxes, iouThreshold);

    // Draw post-NMS boxes on another copy of the image
    cv::Mat imgWithFilteredBoxes = img.clone();
    drawBoxes(imgWithFilteredBoxes, filteredBoxes, cv::Scalar(255, 0, 0)); // Red color for filtered boxes

    // Display the images
    cv::imshow("Original Image with Boxes", imgWithBoxes);
    cv::imshow("Image after NMS", imgWithFilteredBoxes);
    cv::waitKey(0);

    return 0;
}
