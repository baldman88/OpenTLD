#include "TLDTracker.hpp"


TLDTracker::TLDTracker(const int ferns, const int nodes, const double minFutureScale)
    : confidence(1.0), isInitialised(false)
{
    classifier = std::make_shared<Classifier>(ferns, nodes, minFutureScale);
    tracker = std::make_shared<Tracker>(classifier);
}


cv::Rect TLDTracker::getTargetRect(const cv::Mat& frameRGB, const cv::Rect& targetRect)
{
    cv::Mat frame;
    cv::cvtColor(frameRGB, frame, cv::COLOR_RGB2GRAY);
    cv::blur(frame, frame, cv::Size(3, 3));
    cv::Mat integralFrame;
    cv::integral(frame, integralFrame);
    Patch trackedPatch;
    if (isInitialised == false)
    {
        classifier->init(frame, targetRect);
        tracker->init(frame);
        trackedPatch.rect = targetRect;
        isInitialised = true;
    }
    else
    {
        trackedPatch = tracker->track(frame, targetRect);
    }
    return trackedPatch.rect;
}


void TLDTracker::resetTracker()
{
    isInitialised = false;
}
