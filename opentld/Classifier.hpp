#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

#include <vector>
#include <cmath>
#include <functional>
#include <memory>

#include "Fern.hpp"
#include "Concurrent.hpp"


class Classifier
{
public:
    explicit Classifier(const int fernsCount, const int featuresCount, const double minFeatureScale);
    ~Classifier() = default;
    void init(const cv::Mat& frame, const cv::Rect& patch);
    void train(const cv::Mat& frame, const cv::Rect& patch, const int patchClass);
    double classify(const cv::Mat& frame, const cv::Rect& patch) const;
    double getPatchesOverlap(const cv::Rect& patch1, const cv::Rect& patch2) const;
    void trainPositive(const cv::Mat& frame, const cv::Rect& patch);

private:
    std::vector<std::shared_ptr<Fern>> ferns;
    void trainNegative(const cv::Mat& frame, const cv::Rect& patch);
    cv::Mat transform(const cv::Mat& frame, const cv::Point2f& center, const double angle) const;
    cv::Point2f getPatchCenter(const cv::Rect& rect) const;
};

#endif /* CLASSIFIER_HPP */
