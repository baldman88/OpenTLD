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
    const double minOverlap;

private:
    std::vector<std::shared_ptr<Fern>> ferns;
    void trainNegative(const cv::Mat &frame, const cv::Rect &patchRect);
    cv::Mat transform(const cv::Mat &frame, const cv::Point2f &center, const double angle) const;
};

#endif /* CLASSIFIER_HPP */
