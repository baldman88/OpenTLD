#include "Detector.hpp"
#include <iostream>


Detector::Detector(std::shared_ptr<Classifier> &classifier)
: classifier(classifier), patchRectWidth(0), patchRectHeight(0), varianceThreshold(0),
  frameWidth(0), frameHeight(0), minSideSize(16), maxSideSize(120),
  failureCounter(0), failureScaleFactor(0) {}


void Detector::init(const cv::Mat &frame, const cv::Rect &patchRect)
{
    frameWidth = frame.cols;
    frameHeight = frame.rows;
    lastPatchRect = patchRect;
    failureCounter = 0;
    filter.reset();
    setVarianceThreshold(frame, patchRect);
}


void Detector::setVarianceThreshold(const cv::Mat &frame, const cv::Rect &patchRect)
{
    cv::Mat integralFrame;
    cv::Mat squareIntegralFrame;
    cv::integral(frame, integralFrame, squareIntegralFrame);
    varianceThreshold = getPatchVariance(integralFrame, squareIntegralFrame, patchRect) / 2.0;
}


cv::Rect Detector::getCurrentPatchRect(const cv::Rect &patchRect)
{
    cv::Rect currentPatchRect(0, 0, 0, 0);

    if ((patchRect.width >= minSideSize) && (patchRect.height >= minSideSize)
        && (patchRect.width <= maxSideSize) && (patchRect.height <= maxSideSize)
        && (patchRect.x >= 0) && (patchRect.y >= 0)
        && ((patchRect.x + patchRect.width) < frameWidth)
        && ((patchRect.y + patchRect.height) < frameHeight))
    {
        currentPatchRect = patchRect;
    }
    else
    {
        ++failureCounter;
    }

    if ((currentPatchRect.width >= (lastPatchRect.width * 0.85))
        && (currentPatchRect.width <= (lastPatchRect.width * 1.15))
        && (currentPatchRect.height >= (lastPatchRect.height * 0.85))
        && (currentPatchRect.height <= (lastPatchRect.height * 1.15)))
    {
        lastPatchRect = currentPatchRect;
        failureCounter = 0;
    }
    //    else if ((currentPatchRect.width >= (lastPatchRect.width * 0.75))
    //            && (currentPatchRect.width <= (lastPatchRect.width * 1.25))
    //            && (currentPatchRect.height >= (lastPatchRect.height * 0.75))
    //            && (currentPatchRect.height <= (lastPatchRect.height * 1.25)))
    //    {
    //        currentPatchRect = lastPatchRect;
    //    }
    else
    {
        currentPatchRect = lastPatchRect;
    }

    predictedPatchRect = filter.predict(currentPatchRect);
    if ((currentPatchRect.area() == 0)
        && (predictedPatchRect.width >= (lastPatchRect.width * 0.95))
        && (predictedPatchRect.width <= (lastPatchRect.width * 1.05))
        && (predictedPatchRect.height >= (lastPatchRect.height * 0.95))
        && (predictedPatchRect.height <= (lastPatchRect.height * 1.05))
        && (predictedPatchRect.x >= 0)
        && (predictedPatchRect.y >= 0)
        && ((predictedPatchRect.x + predictedPatchRect.width)  < frameWidth)
        && ((predictedPatchRect.y + predictedPatchRect.height) < frameHeight))
    {
        //        currentPatchRect = predictedPatchRect;
    }

    return currentPatchRect;
}


void Detector::detect(const cv::Mat &frame, const cv::Rect &patchRect, std::vector<Patch> &patches)
{
    cv::Rect currentPatchRect = getCurrentPatchRect(patchRect);

    std::set<int> widths;
    std::set<int> heights;

    double minScale = 0.975;
    double maxScale = 1.025;
    double scaleStep = 0.025;

//    if ((currentPatchRect.x != patchRect.x)
//        || (currentPatchRect.y != patchRect.y)
//        || (currentPatchRect.width != patchRect.width)
//        || (currentPatchRect.height != patchRect.height))
//    {
//        minScale = 0.9;
//        maxScale = 1.1;
//    }

    for (double scale = minScale; scale <= maxScale; scale += scaleStep)
    {
        widths.insert(static_cast<int>(round(lastPatchRect.width * scale)));
        heights.insert(static_cast<int>(round(lastPatchRect.height * scale)));
    }

    double stepDevider = 20.0;
    currentPatchRectCenter = classifier->getRectCenter(currentPatchRect);
    predictedPatchRectCenter = classifier->getRectCenter(predictedPatchRect);

    cv::Mat integralFrame;
    cv::Mat squareIntegralFrame;
    cv::integral(frame, integralFrame, squareIntegralFrame);
    //    int xStart = std::max((currentPatchRectCenter.x - (currentPatchRect.width / 2) - currentPatchRect.width), 0);
    //    int yStart = std::max((currentPatchRectCenter.y - (currentPatchRect.height / 2) - currentPatchRect.height), 0);
    //    int xStop = std::min((frameWidth - currentPatchRect.width),
    //                         (currentPatchRectCenter.x - (currentPatchRect.width / 2) + currentPatchRect.width));
    //    int yStop = std::min((frameHeight - currentPatchRect.height),
    //                         (currentPatchRectCenter.y - (currentPatchRect.height / 2) + currentPatchRect.height));
    //    varianceThreshold = getPatchVariance(integralFrame, squareIntegralFrame,
    //                                         cv::Rect(xStart, yStart, (xStop - xStart), (yStop - yStart)));

    std::vector<cv::Rect> testRects;
    failureScaleFactor = std::max((failureCounter / 20), 1);
    for (auto widthIterator = widths.begin(); widthIterator != widths.end(); ++widthIterator)
    {
        int currentWidth = (*widthIterator);
        int xStep = static_cast<int>(round(currentWidth / stepDevider));
        int xCurrent;
        int xMin;
        int xMax;
        if (failureCounter == 0)
        {
            xCurrent = currentPatchRectCenter.x - (static_cast<int>(round(currentWidth / 2.0)));
            xMin = std::max((xCurrent - static_cast<int>(round(currentWidth / 2.0))), 0);
            xMax = std::min((frameWidth - static_cast<int>(round(currentWidth / 2.0))),
                            (xCurrent + static_cast<int>(round(currentWidth / 2.0))));
        }
        else
        {
            //            xCurrent = ((currentPatchRectCenter.x + predictedPatchRectCenter.x) / 2) - (currentWidth / 2);
            xCurrent = currentPatchRectCenter.x - (static_cast<int>(round(currentWidth / 2.0)));
            xMin = std::max((xCurrent - (static_cast<int>(round(currentWidth / 2.0)) * failureScaleFactor)), 0);
            xMax = std::min((frameWidth - static_cast<int>(round(currentWidth / 2.0))),
                            (xCurrent + (static_cast<int>(round(currentWidth / 2.0)) * failureScaleFactor)));
        }
        for (int x = xMin; x < xMax; x += xStep)
        {
            for (auto heightIterator = heights.begin(); heightIterator != heights.end(); ++heightIterator)
            {
                int currentHeight = (*heightIterator);
                int yStep = static_cast<int>(round(currentHeight / stepDevider));
                int yCurrent;
                int yMin;
                int yMax;
                if (failureCounter == 0)
                {
                    yCurrent = currentPatchRectCenter.y - (static_cast<int>(round(currentHeight / 2.0)));
                    yMin = std::max((yCurrent - static_cast<int>(round(currentHeight / 2.0))), 0);
                    yMax = std::min((frameHeight - static_cast<int>(round(currentHeight / 2.0))),
                                    (yCurrent + static_cast<int>(round(currentHeight / 2.0))));
                }
                else
                {
                    //                    yCurrent = ((currentPatchRectCenter.y + predictedPatchRectCenter.y) / 2) - (currentHeight / 2);
                    yCurrent = currentPatchRectCenter.y - (static_cast<int>(round(currentHeight / 2.0)));
                    yMin = std::max((yCurrent - (static_cast<int>(round(currentHeight / 2.0)) * failureScaleFactor)), 0);
                    yMax = std::min((frameHeight - static_cast<int>(round(currentHeight / 2.0))),
                                    (yCurrent + (static_cast<int>(round(currentHeight / 2.0)) * failureScaleFactor)));
                }
                for (int y = yMin; y < yMax; y += yStep)
                {
                    testRects.push_back(cv::Rect(x, y, currentWidth, currentHeight));
                }
            }
        }
    }
    //    auto end = concurrent::blockingFilter(testRects.begin(), testRects.end(),
    //                                          std::bind(&Detector::checkPatchVariace, this,
    //                                                    integralFrame, squareIntegralFrame, std::placeholders::_1));
    //    testRects.erase(end, testRects.end());
    //    std::cout << "testRects.size() = " << testRects.size() << std::endl;
    patches.resize(testRects.size());
    concurrent::blockingMapped(testRects.begin(), testRects.end(), patches.begin(),
                               std::bind(&Detector::getPatch, this, std::placeholders::_1, integralFrame, currentPatchRect));
    if (patches.size() > 0)
    {
        auto end = concurrent::blockingFilter(patches.begin(), patches.end(),
                                              std::bind(&Detector::checkPatchConformity, this, std::placeholders::_1));
        patches.erase(end, patches.end());
    }
    //    patches.push_back(Patch(predictedPatchRect, 0, false));
}


Patch Detector::getPatch(const cv::Rect &testRect, const cv::Mat &frame, const cv::Rect &patchRect) const
{
    double confidence = classifier->classify(frame, testRect);
    double overlap = 0.0;
    if (patchRect.area() > 0)
    {
        overlap = classifier->getRectsOverlap(patchRect, testRect);
    }
    return Patch(testRect, confidence, overlap);
}


bool Detector::checkPatchConformity(const Patch &patch) const
{
    bool result = false;
    if ((patch.confidence > minimumConfidence) || (patch.overlap > minimumOverlap))
    {
        result = true;
    }
    return result;
}


double Detector::getPatchVariance(const cv::Mat &integralFrame, const cv::Mat &squareIntegralFrame, const cv::Rect &patchRect) const
{
    double variance = 0;
    double area = patchRect.area();
    if (area > 0)
    {
        double mean = (integralFrame.at<int>(cv::Point(patchRect.x, patchRect.y))
            + integralFrame.at<int>(cv::Point(patchRect.x + patchRect.width, patchRect.y + patchRect.height))
            - integralFrame.at<int>(cv::Point(patchRect.x + patchRect.width, patchRect.y))
            - integralFrame.at<int>(cv::Point(patchRect.x, patchRect.y + patchRect.height))) / area;
        double deviance = (squareIntegralFrame.at<double>(cv::Point(patchRect.x, patchRect.y))
            + squareIntegralFrame.at<double>(cv::Point(patchRect.x + patchRect.width, patchRect.y + patchRect.height))
            - squareIntegralFrame.at<double>(cv::Point(patchRect.x + patchRect.width, patchRect.y))
            - squareIntegralFrame.at<double>(cv::Point(patchRect.x, patchRect.y + patchRect.height))) / area;
        variance = deviance - (mean * mean);
    }
    return variance;
}


bool Detector::checkPatchVariace(const cv::Mat &integralFrame, const cv::Mat &squareIntegralFrame, const cv::Rect &patchRect) const
{
    bool result = false;
    if (getPatchVariance(integralFrame, squareIntegralFrame, patchRect) > varianceThreshold)
    {
        result = true;
    }
    return result;
}
