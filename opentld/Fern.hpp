#ifndef FERN_HPP
#define FERN_HPP

#include <vector>
#include <mutex>

#include <opencv2/imgproc/imgproc.hpp>

#include "Feature.hpp"

class Fern
{
public:
    explicit Fern(const int featuresCount, const double minScale);
    ~Fern();
    void train(const cv::Mat& frame, const cv::Rect& patch, const int patchClass);
    float classify(const cv::Mat& frame, const cv::Rect& patch) const;
    void reset();

private:
    std::vector<Feature*> features;
    std::vector<double> posteriors;
    std::vector<int> positives;
    std::vector<int> negatives;
    std::mutex mutex;
    int getLeafIndex(const cv::Mat& frame, const cv::Rect& patch) const;
};

#endif /* FERN_HPP */