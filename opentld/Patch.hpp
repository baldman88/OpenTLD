#ifndef PATCH_HPP
#define PATCH_HPP

#include <opencv2/imgproc/imgproc.hpp>


struct Patch
{
    Patch(const cv::Rect &rect = cv::Rect(0, 0, 0, 0), const double confidence = 0.0, const double overlap = 0.0);
    bool operator<(const Patch &other) const;
    Patch &operator=(const Patch &other);
    cv::Rect rect;
    double confidence;
    double overlap;
};

#endif /* PATCH_HPP */
