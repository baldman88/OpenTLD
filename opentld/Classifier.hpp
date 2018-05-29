#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <vector>
#include <cmath>
#include <functional>
#include <memory>
#include <chrono>
#include <set>

#include "Concurrent.hpp"
#include "Fern.hpp"
#include "Constants.hpp"


class Classifier
{
public:
    explicit Classifier(const int fernsCount, const int featuresCount, const double minFeatureScale, const double maxFeatureScale);
    ~Classifier() = default;
    void init(const cv::Mat &frame, const cv::Rect &patchRect);
    void train(const cv::Mat &frame, const cv::Rect &patchRect, const bool isPositive);
    double classify(const cv::Mat &frame, const cv::Rect &patchRect) const;
    double getRectsOverlap(const cv::Rect &first, const cv::Rect &second) const;
    cv::Point2f getRectCenter(const cv::Rect &rect) const;
    void trainPositive(const cv::Mat &frame, const cv::Rect &patchRect);

private:
    std::vector<std::shared_ptr<Fern>> ferns;
    void trainNegative(const cv::Mat &frame, const cv::Rect &patchRect);
    cv::Mat transform(const cv::Mat &frame, const cv::Point2f &center, const double angle) const;
    cv::Mat flipVertical(const cv::Mat &frame) const;
    cv::Mat flipHorizontal(const cv::Mat &frame) const;
    cv::Mat getIntegralFrame(const cv::Mat &frame) const;
    void calculateScaledSizes(const cv::Rect& patchRect, const cv::Size& frameSize,
                              std::set<int>& widths, std::set<int>& heights) const;
    double getMaxRotateAngle(const cv::Rect& patchRect, const cv::Size& frameSize,
                             const std::set<int>& widths, const std::set<int>& heights) const;
//    void getWarpFrames(const cv::Mat& warpFrame, const std::vector<double>& angles, std::vector<cv::Mat>& warpFrames) const;
    template<typename T>
    void trainOnRange(const cv::Mat &frame, const T first, const T last, const bool isPositive)
    {
        std::for_each(first, last, std::bind(&Classifier::train, this, std::ref(frame), std::placeholders::_1, isPositive));
    }
    template<typename T>
    void getIntegralFramesOnRange(const cv::Mat &frame, const T first, const T last, const bool isPositive)
    {
        std::for_each(first, last, std::bind(&Classifier::getIntegralFrame, this, std::placeholders::_1));
    }
};

#endif /* CLASSIFIER_HPP */
