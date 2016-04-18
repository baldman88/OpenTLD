#include "Detector.hpp"
#include <iostream>


Detector::Detector(std::shared_ptr<Classifier>& classifier)
    : classifier(classifier), patchRectWidth(0), patchRectHeight(0),
      frameWidth(0), frameHeight(0), varianceThreshold(0), minSideSize(20) {}


void Detector::init(const cv::Mat& frame, const cv::Rect& patchRect)
{
    frameWidth = frame.cols;
    frameHeight = frame.rows;
    setPatchRectSize(patchRect);
    setVarianceThreshold(frame, patchRect);
}


std::vector<Patch> Detector::detect(const cv::Mat& frame, const cv::Rect& patchRect) const
{
#ifdef DEBUG
    std::cout << "patchRect(" << patchRect.x << ", " << patchRect.y << ", "
            << patchRect.width << ", " << patchRect.height << ")" << std::endl;
#endif
    std::vector<Patch> patches;
    int width;
    int height;
    double minScale = 0.9;
    double maxScale = 1.1;
    double scaleStep = 0.05;
    if ((patchRect.width >= minSideSize) && (patchRect.height >= minSideSize))
    {
        width = patchRect.width;
        height = patchRect.height;
    }
    else
    {
        width = patchRectWidth;
        height = patchRectHeight;
    }
    if ((width < frame.cols) && (height < frame.rows))
    {
        std::vector<cv::Rect> testRects;
        int stepDevider = 20;
        cv::Point patchRectCenter = classifier->getRectCenter(patchRect);
        std::random_device randomDevice;
        std::mt19937 randomEngine(randomDevice());
        for (double scale = minScale; scale <= maxScale; scale += scaleStep)
        {
            int currentWidth = static_cast<int>(round(scale * width));
            if (currentWidth >= minSideSize)
            {
                int xMin;
                int xMax;
                int xStep = static_cast<int>(ceil(currentWidth / stepDevider));
                if (patchRect.area() > 0)
                {
                    int currenterX = patchRectCenter.x - (currentWidth / 2);
                    xMin = std::max((currenterX - currentWidth), 0);
                    xMax = std::min((frameWidth - currentWidth), (currenterX + currentWidth));
                }
                else
                {
                    if ((frameWidth / 3) > currentWidth)
                    {
                        std::uniform_int_distribution<int> distribution(0, (frameWidth - (currentWidth * 3)));
                        xMin = distribution(randomEngine);
                        xMax = xMin + (currentWidth * 2);
                    }
                    else
                    {
                        xMin = 0;
                        xMax = frameWidth - currentWidth;
                    }
                }
                for (int x = xMin; x < xMax; x += xStep)
                {
                    int currentHeight = static_cast<int>(round(scale * height));
                    if (currentHeight >= minSideSize)
                    {
                        int yMin;
                        int yMax;
                        int yStep = static_cast<int>(ceil(currentHeight / stepDevider));
                        if (patchRect.area() > 0)
                        {
                            int currenterY = patchRectCenter.y - (currentHeight / 2);
                            yMin = std::max((currenterY - currentHeight), 0);
                            yMax = std::min((frameHeight - currentHeight), (currenterY + currentHeight));
                        }
                        else
                        {
                            if ((frameHeight / 3) > currentHeight)
                            {
                                std::uniform_int_distribution<int> distribution(0, (frameHeight - (currentHeight * 3)));
                                yMin = distribution(randomEngine);
                                yMax = yMin + (currentHeight * 2);
                            }
                            else
                            {
                                yMin = 0;
                                yMax = frameHeight - currentHeight;
                            }
                        }
                        for (int y = yMin; y < yMax; y += yStep)
                        {
                            if (((x + currentWidth) >= 640) || ((y + currentHeight) >= 480))
                            {
                                std::cout << "We have a problem!" << std::endl;
                            }
                            testRects.push_back(cv::Rect(x, y, currentWidth, currentHeight));
                        }
                    }
                }
            }
        }
        if (testRects.size() > 0)
        {
            cv::Mat integralFrame;
            cv::Mat squareIntegralFrame;
            cv::integral(frame, integralFrame, squareIntegralFrame);
            auto end = concurrent::blockingFilter(testRects.begin(), testRects.end(),
                                                  std::bind(&Detector::checkPatchVariace, this,
                                                            integralFrame, squareIntegralFrame, std::placeholders::_1));
            testRects.erase(end, testRects.end());
            patches.resize(testRects.size());
            concurrent::blockingMapped(testRects.begin(), testRects.end(), patches.begin(),
                                       std::bind(&Detector::getPatch, this, std::placeholders::_1, integralFrame, patchRect));
        }
    }
    if (patches.size() > 0)
    {
        auto end = concurrent::blockingFilter(patches.begin(), patches.end(),
                                              std::bind(&Detector::checkPatchConformity, this, std::placeholders::_1));
        patches.erase(end, patches.end());
    }
    return patches;
}


double Detector::getPatchVariance(const cv::Mat& integralFrame, const cv::Mat& squareIntegralFrame, const cv::Rect& patchRect) const
{
    double variance = 0;
    double area = patchRect.area();
    if (area > 0)
    {
        double mean = (integralFrame.at<int>(cv::Point(patchRect.x, patchRect.y)) +
                      integralFrame.at<int>(cv::Point(patchRect.x + patchRect.width, patchRect.y + patchRect.height)) -
                      integralFrame.at<int>(cv::Point(patchRect.x + patchRect.width, patchRect.y)) -
                      integralFrame.at<int>(cv::Point(patchRect.x, patchRect.y + patchRect.height))) / area;
        double deviance = (squareIntegralFrame.at<float>(cv::Point(patchRect.x, patchRect.y)) +
                          squareIntegralFrame.at<float>(cv::Point(patchRect.x + patchRect.width, patchRect.y + patchRect.height)) -
                          squareIntegralFrame.at<float>(cv::Point(patchRect.x + patchRect.width, patchRect.y)) -
                          squareIntegralFrame.at<float>(cv::Point(patchRect.x, patchRect.y + patchRect.height))) / area;
        variance = deviance - (mean * mean);
    }
    return variance;
}


Patch Detector::getPatch(const cv::Rect& testRect, const cv::Mat& frame, const cv::Rect& patchRect) const
{
    double confidence = classifier->classify(frame, testRect);
    int overlap = 0;
    if ((patchRect.area() != 0) && (classifier->getRectsOverlap(patchRect, testRect) > classifier->minOverlap))
    {
        overlap = 1;
    }
    return Patch(testRect, confidence, overlap);
}


bool Detector::checkPatchConformity(const Patch& patch) const
{
    bool result = false;
    if ((patch.confidence > 0.6f) || (patch.isOverlaps == true))
    {
        result = true;
    }
    return result;
}


bool Detector::checkPatchVariace(const cv::Mat& integralFrame, const cv::Mat& squareIntegralFrame, const cv::Rect& patchRect) const
{
    bool result = false;
    if (getPatchVariance(integralFrame, squareIntegralFrame, patchRect) > varianceThreshold)
    {
        result = true;
    }
    return result;
}


void Detector::setVarianceThreshold(const cv::Mat& frame, const cv::Rect& patchRect)
{
    cv::Mat integralFrame;
    cv::Mat squareIntegralFrame;
    cv::integral(frame, integralFrame, squareIntegralFrame);
    varianceThreshold = getPatchVariance(integralFrame, squareIntegralFrame, patchRect);
}


void Detector::setPatchRectSize(const cv::Rect& patchRect)
{
    patchRectWidth = patchRect.width;
    patchRectHeight = patchRect.height;
}
