#ifndef IMAGESTREAMPROCESSOR_H
#define IMAGESTREAMPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <future>
#include <vector>
#include "ImageSlicer.h"
#include "NMS.h"
#include "YourYoloModel.h"  // Replace with your actual YOLO model header

class ImageStreamProcessor {
public:
    ImageStreamProcessor(YourYoloModel& model, int slice_height, int slice_width,
                         float overlap_height_ratio, float overlap_width_ratio)
    : model_(model), slicer_(slice_height, slice_width, overlap_height_ratio, overlap_width_ratio) {}

    void processStream(cv::VideoCapture& stream) {
        cv::Mat frame;
        while (stream.read(frame)) {
            auto sliceRegions = slicer_.calculateSliceRegions(frame.rows, frame.cols);
            auto boxes = predictBoundingBoxes(frame, sliceRegions);
            auto mappedBoxes = mapBoundingBoxesToOriginal(boxes);
            auto finalBoxes = applyNMS(mappedBoxes);

            // Further processing or display the finalBoxes on the frame
        }
    }

private:
    YourYoloModel& model_;
    ImageSlicer slicer_;
    NMS nms_;

    std::vector<BoundingBox> predictBoundingBoxes(const cv::Mat& image, const std::vector<std::pair<cv::Rect, int>>& sliceRegions) {
        std::vector<BoundingBox> boxes;
        for (const auto& [region, index] : sliceRegions) {
            cv::Mat slice = image(region);
            auto sliceBoxes = model_.predict(slice);

            // Assign the slice index to each bounding box
            for (auto& box : sliceBoxes) {
                box.sliceIndex = index;
            }

            boxes.insert(boxes.end(), sliceBoxes.begin(), sliceBoxes.end());
        }
        return boxes;
    }

    std::vector<BoundingBox> mapBoxesToOriginal(const std::vector<BoundingBox>& boxes, int sliceIndex, int sliceWidth, int sliceHeight) {
        std::vector<BoundingBox> mappedBoxes;

        return mappedBoxes;
    }

    std::vector<BoundingBox> applyNMS(std::vector<BoundingBox>& boxes) {
        return nms_.nonMaximumSupression(boxes, 0.5);  // Example IoU threshold
    }
};

#endif //IMAGESTREAMPROCESSOR_H
