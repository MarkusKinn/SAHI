#ifndef NMS_H
#define NMS_H

#include <algorithm>
#include <vector>


struct BoundingBox {
    float x, y, w, h, score;
};

class NMS {
public:
    float IoU(const BoundingBox& a, const BoundingBox& b) {

        float x1 = std::max(a.x, b.x);
        float y1 = std::max(a.y, b.y);
        float x2 = std::min(a.x + a.w, b.x + b.w);
        float y2 = std::min(a.y + a.h, b.y + b.h);

        float intersectionWidth = std::max(0.0f, x2 - x1);
        float intersectionHeight = std::max(0.0f, y2 - y1);


        float intersectionArea = intersectionWidth * intersectionHeight;

        float areaA = a.w * a.h;
        float areaB = b.w * b.h;

        float unionArea = areaA + areaB - intersectionArea;

        return intersectionArea / unionArea;
    }

    std::vector<BoundingBox> nonMaximumSupression(std::vector<BoundingBox>& boxes, float iouThreshold) {
        std::vector<BoundingBox> result;

        // Sort boxes by score in descending order
        std::sort(boxes.begin(), boxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
                return a.score > b.score;
            });

        while (!boxes.empty()) {
            // Take the box with the highest score
            BoundingBox current = boxes.front();
            result.push_back(current);
            boxes.erase(boxes.begin());

            // Compare with the rest of the boxes
            for (auto it = boxes.begin(); it != boxes.end();) {
                if (IoU(current, *it) > iouThreshold) {
                    // If IoU is above the threshold, remove the box
                    it = boxes.erase(it);
                } else {
                    ++it;
                }
            }
        }
        return result;
    }
};


#endif //NMS_H
