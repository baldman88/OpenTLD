#ifndef FERN_HPP
#define FERN_HPP

#include <vector>
#include <mutex>
#include <memory>

#include <opencv2/imgproc/imgproc.hpp>

#include "Feature.hpp"

class Fern
{
public:
    explicit Fern(const int featuresCount, const double minScale);
    ~Fern() = default;
    void train(const cv::Mat& frame, const cv::Rect& patchRect, const bool isPositive);
    double classify(const cv::Mat& frame, const cv::Rect& patchRect) const;
    void reset();

private:
    std::vector<std::shared_ptr<Feature>> features;
    std::vector<double> posteriors;
    std::vector<int> positives;
    std::vector<int> negatives;
    std::mutex mutex;
    int getLeafIndex(const cv::Mat& frame, const cv::Rect& patchRect) const;
};

#endif /* FERN_HPP */
