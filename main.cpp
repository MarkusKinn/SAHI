#include "NMS.h"
#include <iostream>

int main() {
    // Create an NMS object
    NMS nms;

    // Create sample bounding boxes
    std::vector<BoundingBox> boxes = {
        {10, 10, 100, 100, 0.9}, // x, y, w, h, score
        {15, 15, 100, 100, 0.8},
        {20, 20, 100, 100, 0.7},
        {100, 100, 200, 200, 0.95},
        {200, 200, 50, 50, 0.6}
    };

    // Apply non-maximum suppression
    float iouThreshold = 0.5; // You can adjust this threshold as needed
    std::vector<BoundingBox> filteredBoxes = nms.nonMaximumSupression(boxes, iouThreshold);

    // Print the results
    std::cout << "Filtered Boxes:" << std::endl;
    for (const auto& box : filteredBoxes) {
        std::cout << "Box at (" << box.x << ", " << box.y << ") with width " << box.w
                  << " and height " << box.h << " (score: " << box.score << ")" << std::endl;
    }

    return 0;
}
