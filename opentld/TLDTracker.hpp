#ifndef TLDTRACKER_HPP
#define TLDTRACKER_HPP

#include <vector>
#include <memory>

#include <opencv2/imgproc/imgproc.hpp>

#include "Classifier.hpp"
#include "Patch.hpp"
#include "Tracker.hpp"


class TLDTracker
{
public:
    TLDTracker(const int ferns = 10, const int nodes = 8, const double minFutureScale = 0.3);
    ~TLDTracker() = default;
    cv::Rect getTargetRect(const cv::Mat& frameRGB, const cv::Rect& targetRect);
    void resetTracker();

private:
    std::shared_ptr<Classifier> classifier;
    std::shared_ptr<Tracker> tracker;
    double confidence;
    bool isInitialised;
};

#endif /* TLDTRACKER_HPP */