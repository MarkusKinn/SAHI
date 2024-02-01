#include "SAHI.h"

SAHI::SAHI(int slice_height, int slice_width, float overlap_height_ratio, float overlap_width_ratio)
    : slice_height_(slice_height), slice_width_(slice_width),
      overlap_height_ratio_(overlap_height_ratio), overlap_width_ratio_(overlap_width_ratio) {}

std::vector<std::pair<cv::Rect, int>> SAHI::calculateSliceRegions(int image_height, int image_width) {
    std::vector<std::pair<cv::Rect, int>> regions;

    int index = 0;

    image_width_ = image_width;
    image_height_ = image_height;

    int step_height = slice_height_ - static_cast<int>(slice_height_ * overlap_height_ratio_);
    int step_width = slice_width_ - static_cast<int>(slice_width_ * overlap_width_ratio_);

    for (int y = 0; y < image_height; y += step_height) {
        for (int x = 0; x < image_width; x += step_width) {
            int width = slice_width_;
            int height = slice_height_;

            // Adjust width and height for slices at the edges of the image
            if (x + width > image_width) width = image_width - x;
            if (y + height > image_height) height = image_height - y;

            regions.emplace_back(cv::Rect(x, y, width, height), index++);
        }
    }
    return regions;
}

cv::Rect SAHI::mapToOriginal(const BoundingBox& boundingBox, const cv::Rect& sliceRegion) {

}




std::vector<BoundingBox> SAHI::nonMaximumSuppression(std::vector<BoundingBox> &boxes, float iouThreshold) {
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


void SAHI::slice(const cv::Mat& image, const std::function<void(const cv::Mat&, int)>& processSlice) {
    std::vector<std::pair<cv::Rect, int>> regions = calculateSliceRegions(image.rows, image.cols);

    // Parallelize this loop with OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto& [region, index] = regions[i];  // Structured binding
        cv::Mat slice = image(region); // Without cloning if processSlice doesn't modify the slice

        processSlice(slice, index);  // Pass slice and its index
    }
}

float SAHI::IoU(const BoundingBox& a, const BoundingBox& b) {

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
