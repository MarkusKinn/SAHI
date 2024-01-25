#include <future>
#include <vector>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "ImageSlicer.h"

ImageSlicer::ImageSlicer(int slice_height, int slice_width, float overlap_height_ratio, float overlap_width_ratio)
    : slice_height_(slice_height), slice_width_(slice_width),
      overlap_height_ratio_(overlap_height_ratio), overlap_width_ratio_(overlap_width_ratio) {}

std::vector<cv::Rect> ImageSlicer::calculateSliceRegions(int image_height, int image_width) {
    std::vector<cv::Rect> regions;

    int step_height = slice_height_ - static_cast<int>(slice_height_ * overlap_height_ratio_);
    int step_width = slice_width_ - static_cast<int>(slice_width_ * overlap_width_ratio_);

    for (int y = 0; y < image_height; y += step_height) {
        for (int x = 0; x < image_width; x += step_width) {
            int width = slice_width_;
            int height = slice_height_;

            // Adjust width and height for slices at the edges of the image
            if (x + width > image_width) width = image_width - x;
            if (y + height > image_height) height = image_height - y;

            regions.push_back(cv::Rect(x, y, width, height));
        }
    }

    return regions;
}


void ImageSlicer::slice(const cv::Mat& image, std::function<void(const cv::Mat&)> processSlice) {
    std::vector<cv::Rect> regions = calculateSliceRegions(image.rows, image.cols);

    // Parallelize this loop with OpenMP
#pragma omp parallel for
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto& region = regions[i];
        cv::Mat slice = image(region); // Without cloning if processSlice doesn't modify the slice

        // OpenMP manages the creation and destruction of threads efficiently
        processSlice(slice);
    }
}

