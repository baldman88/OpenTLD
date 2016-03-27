#ifndef FEATURE_HPP
#define FEATURE_HPP

#include <random>

#include <opencv2/imgproc/imgproc.hpp>

class Feature
{
public:
    Feature(const double minScale);
    int test(const cv::Mat& frame, const cv::Rect& patchRect);

private:
    double scaleX;
    double scaleY;
    double scaleW;
    double scaleH;
    int sumRect(const cv::Mat& frame, const cv::Rect& patchRect);
};

#endif /* FEATURE_HPP */
