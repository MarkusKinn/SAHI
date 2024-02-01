#include "SAHI.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <functional>

BoundingBox generateRandomBoundingBox(const cv::Rect& region) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.1, 0.9);

    // Adjusted to ensure bounding box fits within the slice
    float maxWidth = std::max(1.0f, region.width / 2.0f);
    float maxHeight = std::max(1.0f, region.height / 2.0f);

    float w = dis(gen) * maxWidth;
    float h = dis(gen) * maxHeight;
    float x = region.x + dis(gen) * (region.width - w); // Adjust x to fit the width
    float y = region.y + dis(gen) * (region.height - h); // Adjust y to fit the height
    float score = dis(gen); // Random score between 0.1 and 0.9

    return BoundingBox(x, y, w, h, score, 0); // sliceIndex will be set later
}

int main() {
    // Load an image (replace with your image path)
    cv::Mat image = cv::imread("test_640.jpg");
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }
    cv::imshow("Original Image", image);
    cv::waitKey(0); // Wait for a key press

    // Initialize SAHI with desired parameters
    int sliceHeight = 160, sliceWidth = 160;
    float overlapHeightRatio = 0.2f, overlapWidthRatio = 0.2f;
    SAHI sahi(sliceHeight, sliceWidth, overlapHeightRatio, overlapWidthRatio);

    std::vector<BoundingBox> allBoxes;

    // Process each slice and display the slice with generated bounding boxes
    sahi.slice(image, [&](const cv::Mat& slice, int index) {
        if (slice.empty()) {
            std::cerr << "Error: Slice is empty." << std::endl;
            return;
        }

        // Generate and display bounding boxes on the slice
        for (int i = 0; i < 3; ++i) {
            BoundingBox box = generateRandomBoundingBox(cv::Rect(0, 0, slice.cols, slice.rows));
            box.sliceIndex = index;
            allBoxes.push_back(box);
        }
    });

    std::vector<BoundingBox> mappedBoxes; // Create a vector to store mapped bounding boxes

    cv::Mat imageWithBoxes = image.clone();
    for (auto& box : allBoxes) {
        auto region = sahi.calculateSliceRegions(image.rows, image.cols)[box.sliceIndex];
        auto mappedBox = sahi.mapToOriginal(box, region.first);
        mappedBoxes.push_back(mappedBox); // Add mappedBox to the mappedBoxes vector
        cv::rectangle(imageWithBoxes, cv::Rect(mappedBox.x, mappedBox.y, mappedBox.w, mappedBox.h), cv::Scalar(255, 0, 0), 2);
    }

    cv::imshow("Image with Mapped Boxes", imageWithBoxes);
    cv::waitKey(0);

    // Apply Non-Maximum Suppression
    float iouThreshold = 0.00001f;
    auto finalBoxes = sahi.nonMaximumSuppression(mappedBoxes, iouThreshold);

    // Display the final image with bounding boxes after Non-Maximum Suppression
    cv::Mat finalImage = image.clone();
    for (const auto& box : finalBoxes) {
        cv::rectangle(finalImage, cv::Rect(box.x, box.y, box.w, box.h), cv::Scalar(0, 0, 255), 2);
    }
    cv::imshow("Final Image after NMS", finalImage);
    cv::waitKey(0);

    // Close all OpenCV windows before exiting
    cv::destroyAllWindows();

    return 0;
}
