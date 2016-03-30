#include "TLDTracker.hpp"


TLDTracker::TLDTracker(const int ferns, const int nodes, const double minFutureScale)
    : confidence(1.0), isInitialised(false), trackingConfidence(0.75),
      reinitConfidence(0.75), learningConfidence(0.85)
{
    classifier = std::make_shared<Classifier>(ferns, nodes, minFutureScale);
    detector = std::make_shared<Detector>(classifier);
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
        if (isInitialised == false) {
            classifier->init(frame, targetRect);
            detector->init(frame, targetRect);
            tracker->init(frame);
            confidence = 1.0;
            trackedPatch.rect = targetRect;
            isInitialised = true;
        } else {
            std::vector<Patch> detectedPatches;
            if ((confidence > trackingConfidence) && (targetRect.area() > 0))
            {
                trackedPatch = tracker->track(frame, targetRect);
                detectedPatches = detector->detect(frame, trackedPatch.rect);
            }
            else
            {
                detectedPatches = detector->detect(frame, cv::Rect(0, 0, 0, 0));
            }
            float maxDetectedConfidence = 0.0f;
            int maxDetectedConfidenceIndex = -1;
            for (size_t i = 0; i < detectedPatches.size(); ++i)
            {
                float detectedConfidence = detectedPatches.at(i).confidence;
                if ((detectedPatches.at(i).isOverlaps == true) || ((targetRect.area() == 0) && (detectedConfidence > 0.95)))
                {

                    if (detectedConfidence > maxDetectedConfidence)
                    {
                        maxDetectedConfidence = detectedConfidence;
                        maxDetectedConfidenceIndex = i;
                    }
                }
            }
            if ((trackedPatch.confidence < reinitConfidence) && (maxDetectedConfidence >= reinitConfidence))
            {
                trackedPatch = detectedPatches.at(maxDetectedConfidenceIndex);
    //            detector->setVarianceThreshold(frame, trackedPatch.rect);
                detector->setPatchRectSize(trackedPatch.rect);
            }
            if (targetRect.area() > 0)
            {
                if (trackedPatch.confidence >= learningConfidence)
                {
                    classifier->trainPositive(frame, trackedPatch.rect);
                }
                for (size_t i = 0; i < detectedPatches.size(); ++i)
                {
                    if ((detectedPatches.at(i).confidence < reinitConfidence) || (detectedPatches.at(i).isOverlaps == true))
                    {
                        classifier->train(integralFrame, detectedPatches.at(i).rect, 0);
                    }
                }
            }
            confidence = trackedPatch.confidence;
        }
        return trackedPatch.rect;
}


void TLDTracker::resetTracker()
{
    isInitialised = false;
}
