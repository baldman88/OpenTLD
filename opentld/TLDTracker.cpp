#include "TLDTracker.hpp"


TLDTracker::TLDTracker(const int ferns, const int nodes, const double minFeatureScale, const double maxFeatureScale)
: lastConfidence(1.0), isInitialised(false)
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
            if ((patch.rect.width >= static_cast<int>(round(targetRect.width * 0.85)))
                && (patch.rect.width <= static_cast<int>(round(targetRect.width * 1.15)))
                && (patch.rect.height >= static_cast<int>(round(targetRect.height * 0.85)))
                && (patch.rect.height <= static_cast<int>(round(targetRect.height * 1.15))))
            {
                trackedPatch = patch;
            }
        }

#ifdef USE_DEBUG_INFO
        auto start = std::chrono::high_resolution_clock::now();
#endif // USE_LOGGING

        detector->detect(frame, targetRect, detectedPatches);

#ifdef USE_DEBUG_INFO
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Detector elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
#endif // USE_LOGGING

        float maxDetectedConfidence = 0.0;
        int maxDetectedConfidenceIndex = -1;
        for (size_t i = 0; i < detectedPatches.size(); ++i)
        {
            float confidence = detectedPatches.at(i).confidence;
            if (confidence > detectionConfidence)
            {

                if ((confidence > maxDetectedConfidence)
                    /*&& (((targetRect.area() > 0)
                        && (detectedPatches.at(i).rect.width >= static_cast<int>(round(targetRect.width * 0.9)))
                        && (detectedPatches.at(i).rect.width <= static_cast<int>(round(targetRect.width * 1.1)))
                        && (detectedPatches.at(i).rect.height >= static_cast<int>(round(targetRect.height * 0.9)))
                        && (detectedPatches.at(i).rect.height <= static_cast<int>(round(targetRect.height * 1.1))))
                        || (targetRect.area() == 0))*/)
                {
                    maxDetectedConfidence = confidence;
                    maxDetectedConfidenceIndex = i;
                }
            }
        }

#ifdef USE_DEBUG_INFO
        std::cout << "maxDetectedConfidence = " << maxDetectedConfidence << std::endl;
#endif // USE_LOGGING

        if (((trackedPatch.confidence < reinitConfidence)
            && (maxDetectedConfidence >= reinitConfidence))
            || (trackedPatch.confidence < maxDetectedConfidence))
        {
            trackedPatch = detectedPatches.at(maxDetectedConfidenceIndex);
        }
        if (targetRect.area() > 0)
        {
            if ((trackedPatch.confidence >= learningConfidence)
                 && (trackedPatch.overlap > conformityOverlap)
                 /*&& ((targetRect.area() > 0)
                     && (trackedPatch.rect.width >= static_cast<int>(round(targetRect.width * 0.9)))
                     && (trackedPatch.rect.width <= static_cast<int>(round(targetRect.width * 1.1)))
                     && (trackedPatch.rect.height >= static_cast<int>(round(targetRect.height * 0.9)))
                     && (trackedPatch.rect.height <= static_cast<int>(round(targetRect.height * 1.1))))*/)
            {
                classifier->trainPositive(frame, trackedPatch.rect);
            }
            for (size_t i = 0; i < detectedPatches.size(); ++i)
            {
                if ((detectedPatches.at(i).confidence < negativeConfidence)
                    /*|| ((targetRect.area() > 0)
                        && (detectedPatches.at(i).overlap < conformityOverlap))
                    || ((trackedPatch.rect.area() < static_cast<int>(round(targetRect.area() * 0.85)))
                        || (trackedPatch.rect.area() > static_cast<int>(round(targetRect.area() * 1.15))))*/)
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
