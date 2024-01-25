#ifndef IMAGE_SLICER_H
#define IMAGE_SLICER_H

#include <opencv2/opencv.hpp>
#include <functional>
#include <vector>

class ImageSlicer {
public:
    ImageSlicer(int slice_height, int slice_width, float overlap_height_ratio, float overlap_width_ratio);

    std::vector<std::pair<cv::Rect, int>> calculateSliceRegions(int image_height, int image_width);
    void slice(const cv::Mat& image, const std::function<void(const cv::Mat&, int)>& processSlice);

private:
    int slice_height_;
    int slice_width_;
    float overlap_height_ratio_;
    float overlap_width_ratio_;
};

#endif // IMAGE_SLICER_H
