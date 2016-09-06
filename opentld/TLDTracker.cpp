#include "TLDTracker.hpp"


TLDTracker::TLDTracker(const int ferns, const int nodes, const double minFeatureScale, const double maxFeatureScale)
: lastConfidence(1.0), isInitialised(false), trackingConfidence(0.85),
  reinitConfidence(0.85), learningConfidence(0.9), minimumConfidence(0.7), detectedConfidence(0.9)
{
    classifier = std::make_shared<Classifier>(ferns, nodes, minFeatureScale, maxFeatureScale);
    detector = std::make_shared<Detector>(classifier);
    tracker = std::make_shared<Tracker>(classifier);
}


cv::Rect TLDTracker::getTargetRect(cv::Mat &frameRGB, const cv::Rect &targetRect)
{
    cv::Mat frame;
    cv::cvtColor(frameRGB, frame, cv::COLOR_RGB2GRAY);
//    frame.convertTo(frame, -1, 2, 3);
    cv::blur(frame, frame, cv::Size(3, 3));
    cv::Mat integralFrame;
    cv::integral(frame, integralFrame);
    Patch trackedPatch;
    if (isInitialised == false) {
        classifier->init(frame, targetRect);
        detector->init(frame, targetRect);
        tracker->init(frame);
        lastConfidence = 1.0;
        trackedPatch.rect = targetRect;
        isInitialised = true;
    } else {
        std::vector<Patch> detectedPatches;
        if ((lastConfidence > trackingConfidence) && (targetRect.area() > 0))
        {
            Patch patch = tracker->track(frame, targetRect);
            if ((patch.rect.width >= (targetRect.width * 0.9))
                            && (patch.rect.width <= (targetRect.width * 1.1))
                            && (patch.rect.height >= (targetRect.height * 0.9))
                            && (patch.rect.height <= (targetRect.height * 1.1)))
            {
                trackedPatch = patch;
            }
            detector->detect(frame, trackedPatch.rect, detectedPatches);
        }
        else
        {
            detector->detect(frame, targetRect, detectedPatches);
        }
        float maxDetectedConfidence = 0.0f;
        int maxDetectedConfidenceIndex = -1;
        for (size_t i = 0; i < detectedPatches.size(); ++i)
        {
            float confidence = detectedPatches.at(i).confidence;
            if (confidence > detectedConfidence)
            {

                if ((confidence > maxDetectedConfidence)
                    && (detectedPatches.at(i).rect.width >= (targetRect.width * 0.9))
                    && (detectedPatches.at(i).rect.width <= (targetRect.width * 1.1))
                    && (detectedPatches.at(i).rect.height >= (targetRect.height * 0.9))
                    && (detectedPatches.at(i).rect.height <= (targetRect.height * 1.1)))
                {
                    maxDetectedConfidence = confidence;
                    maxDetectedConfidenceIndex = i;
                }
            }
        }
        std::cout << "maxDetectedConfidence = " << maxDetectedConfidence << std::endl;
        if (((trackedPatch.confidence < reinitConfidence) && (maxDetectedConfidence >= reinitConfidence))
                        || (trackedPatch.confidence < maxDetectedConfidence))
        {
            trackedPatch = detectedPatches.at(maxDetectedConfidenceIndex);
        }
        if (targetRect.area() > 0)
        {
            if (trackedPatch.confidence >= learningConfidence)
            {
                classifier->trainPositive(frame, trackedPatch.rect);
            }
            for (size_t i = 0; i < detectedPatches.size(); ++i)
            {
                if (detectedPatches.at(i).confidence < minimumConfidence)
                {
                    classifier->train(integralFrame, detectedPatches.at(i).rect, false);
                }
            }
        }
        lastConfidence = trackedPatch.confidence;
    }
    return trackedPatch.rect;
}


void TLDTracker::resetTracker()
{
    isInitialised = false;
}
