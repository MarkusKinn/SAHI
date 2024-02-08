#include "SAHI.h"

SAHI::SAHI(int slice_height, int slice_width, float overlap_height_ratio, float overlap_width_ratio)
    : slice_height_(slice_height), slice_width_(slice_width),
      overlap_height_ratio_(overlap_height_ratio), overlap_width_ratio_(overlap_width_ratio) {}

std::vector<std::pair<cv::Rect, int>> SAHI::calculateSliceRegions(int image_height, int image_width) {
    int step_height = slice_height_ - static_cast<int>(slice_height_ * overlap_height_ratio_);
    int step_width = slice_width_ - static_cast<int>(slice_width_ * overlap_width_ratio_);

    int num_regions_x = (image_width + step_width - 1) / step_width; // ceil division
    int num_regions_y = (image_height + step_height - 1) / step_height; // ceil division

    std::vector<std::pair<cv::Rect, int>> regions;
    regions.reserve(num_regions_x * num_regions_y);

    #pragma omp parallel for collapse(2) // Parallelize both outer and inner loop
    for (int y = 0; y < image_height; y += step_height) {
        for (int x = 0; x < image_width; x += step_width) {
            int height = std::min(slice_height_, image_height - y);
            int width = std::min(slice_width_, image_width - x);
            int index = (y / step_height) * num_regions_x + (x / step_width);

            #pragma omp critical
            regions.emplace_back(cv::Rect(x, y, width, height), index);
        }
    }

    return regions;
}

BoundingBox SAHI::mapToOriginal(BoundingBoxIndex& boundingBox, const cv::Rect& sliceRegion) {
    return BoundingBox(boundingBox.x + sliceRegion.x, boundingBox.y + sliceRegion.y,
                       boundingBox.h, boundingBox.w, boundingBox.score);
}

void SAHI::slice(const cv::Mat& image, const std::function<void(const cv::Mat&, int)>& processSlice) {
    std::vector<std::pair<cv::Rect, int>> regions = calculateSliceRegions(image.rows, image.cols);

    // Parallelize this loop with OpenMP
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto& [region, index] = regions[i];  // Structured binding
        cv::Mat slice = image(region); // Without cloning if processSlice doesn't modify the slice

        processSlice(slice, index);  // Pass slice and its index
    }
}