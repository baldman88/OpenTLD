#ifndef TLDTRACKER_HPP
#define TLDTRACKER_HPP

#include <vector>
#include <memory>

#include <opencv2/imgproc/imgproc.hpp>

#include "Classifier.hpp"
#include "Patch.hpp"
#include "Tracker.hpp"
#include "Detector.hpp"
#include "Constants.hpp"


class TLDTracker
{
public:
    TLDTracker(const int ferns = 8, const int nodes = 8, const double minFeatureScale = 0.2, const double maxFeatureScale = 0.5);
    ~TLDTracker() = default;
    cv::Rect getTargetRect(cv::Mat &frameRGB, const cv::Rect &targetRect);
    void resetTracker();

private:
    std::shared_ptr<Classifier> classifier;
    std::shared_ptr<Detector> detector;
    std::shared_ptr<Tracker> tracker;
    double lastConfidence;
    bool isInitialised;
};

#endif /* TLDTRACKER_HPP */
