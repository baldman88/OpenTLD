#include "Classifier.hpp"
#include <iostream>


Classifier::Classifier(const int fernsCount, const int featuresCount, const double minFeatureScale)
    : minOverlap(0.6)
{
    for (int fern = 0; fern < fernsCount; ++fern)
    {
        ferns.push_back(std::make_shared<Fern>(featuresCount, minFeatureScale));
    }
}


void Classifier::train(const cv::Mat& frame, const cv::Rect& patchRect, const bool isPositive)
{
    for (auto fern: ferns)
    {
        fern->train(frame, patchRect, isPositive);
    }
}


void Classifier::trainNegative(const cv::Mat& frame, const cv::Rect& patchRect)
{
    double minScale = 0.5;
    double maxScale = 1.5;
    double scaleStep = 0.25;
    for (double scale = minScale; scale <= maxScale; scale += scaleStep)
    {
        int xMin = 0;
        int currentWidth = static_cast<int>(scale * patchRect.width);
        int xMax = frame.cols - currentWidth;
        int xStep = 10;
        for (int x = xMin; x < xMax; x += xStep)
        {
            int yMin = 0;
            int currentHeight = static_cast<int>(scale * patchRect.height);
            int yMax = frame.rows - currentHeight;
            int yStep = 10;
            for (int y = yMin; y < yMax; y += yStep)
            {
                cv::Rect negativePatchRect(x, y, currentWidth, currentHeight);
                if (getRectsOverlap(patchRect, negativePatchRect) < minOverlap)
                {
                    train(frame, negativePatchRect, 0);
                }
            }
        }
    }
}


//void Classifier::trainPositive(const cv::Mat& frame, const cv::Rect& patchRect)
//{
//    cv::Point2f patchRectCenter = getRectCenter(patchRect);
//    double maxAngle;
//    cv::Rect warpFrameRect;
//    /* warpSize takes into account the size of the maximum values of the offset (±3 pixels) and scaling (1.1) */
//    cv::Size warpSize(static_cast<int>(std::ceil(patchRect.width * 1.1 + 6)),
//                      static_cast<int>(std::ceil(patchRect.height * 1.1 + 6)));
//    for (maxAngle = 10.0; maxAngle > 0.0; maxAngle -= 1.0)
//    {
//        warpFrameRect = cv::RotatedRect(patchRectCenter, warpSize, maxAngle).boundingRect();
//        if ((warpFrameRect.tl().x >= 0) && (warpFrameRect.tl().y >= 0)
//            && (warpFrameRect.br().x < frame.cols) && (warpFrameRect.br().y < frame.rows))
//        {
//            break;
//        }
//    }
//    cv::Mat warpFrame = frame(warpFrameRect);
//    cv::Rect warpPatchRect((patchRect.x - warpFrameRect.x), (patchRect.y - warpFrameRect.y), patchRect.width, patchRect.height);
//    std::vector<double> angles;
//    for (double angle = -maxAngle; angle <= maxAngle; angle += 1.0)
//    {
//        angles.push_back(angle);
//    }
//    cv::Point2f warpPatchRectCenter = getRectCenter(warpPatchRect);
//    std::vector<cv::Mat> warpFrames(angles.size());
//    concurrent::blockingMapped(angles.begin(), angles.end(), warpFrames.begin(),
//                               std::bind(&Classifier::transform, this, warpFrame, warpPatchRectCenter, std::placeholders::_1));
//    std::vector<cv::Rect> positivePatches;
//    for (int xOffset = -3; xOffset <= 3; xOffset += 1)
//    {
//        for (int yOffset = -3; yOffset <= 3; yOffset += 1)
//        {
//            for (double scale = 0.9; scale <= 1.1; scale += 0.1)
//            {
//                int width = round(warpPatchRect.width * scale);
//                int height = round(warpPatchRect.height * scale);
//                int x = warpPatchRectCenter.x - (width / 2) + xOffset;
//                int y = warpPatchRectCenter.y - (height / 2) + yOffset;
//                positivePatches.push_back(cv::Rect(x, y, width, height));
//            }
//        }
//    }
//    for (size_t i = 0; i < warpFrames.size(); ++i)
//    {
//        concurrent::blockingMap(positivePatches.begin(), positivePatches.end(),
//                                std::bind(&Classifier::train, this, warpFrames.at(i), std::placeholders::_1, 1));
//    }
//}


void Classifier::trainPositive(const cv::Mat& frame, const cv::Rect& patchRect)
{
    cv::Point2f patchRectCenter = getRectCenter(patchRect);
    for (double scale = 0.9; scale <= 1.1; scale += 0.1)
    {
        cv::Size warpSize(static_cast<int>(std::ceil(patchRect.width * scale + 6)),
                          static_cast<int>(std::ceil(patchRect.height * scale + 6)));
        if (((patchRectCenter.x - (warpSize.width / 2)) >= 0)
                && ((patchRectCenter.x + (warpSize.height / 2)) < frame.cols)
                && ((patchRectCenter.y - (warpSize.height / 2)) >= 0)
                && ((patchRectCenter.y + (warpSize.height / 2)) < frame.rows))
        {
            double maxAngle;
            cv::Rect warpFrameRect;
            for (maxAngle = 1.0; maxAngle < 10.0; maxAngle += 1.0)
            {
                warpFrameRect = cv::RotatedRect(patchRectCenter, warpSize, maxAngle).boundingRect();
                if ((warpFrameRect.tl().x < 0) || (warpFrameRect.tl().y < 0)
                        || (warpFrameRect.br().x >= frame.cols) || (warpFrameRect.br().y >= frame.rows))
                {
                    break;
                }
            }
            maxAngle -= 1.0;
            warpFrameRect = cv::RotatedRect(patchRectCenter, warpSize, maxAngle).boundingRect();
            cv::Mat warpFrame = frame(warpFrameRect);
            cv::Rect warpPatchRect((patchRect.x - warpFrameRect.x), (patchRect.y - warpFrameRect.y), patchRect.width, patchRect.height);
            std::vector<double> angles;
            for (double angle = -maxAngle; angle <= maxAngle; angle += 1.0)
            {
                angles.push_back(angle);
            }
            cv::Point2f warpPatchRectCenter = getRectCenter(warpPatchRect);
            std::vector<cv::Mat> warpFrames(angles.size());
            concurrent::blockingMapped(angles.begin(), angles.end(), warpFrames.begin(),
                                       std::bind(&Classifier::transform, this, warpFrame, warpPatchRectCenter, std::placeholders::_1));
            std::vector<cv::Rect> positivePatches;
            for (int xOffset = -3; xOffset <= 3; xOffset += 1)
            {
                for (int yOffset = -3; yOffset <= 3; yOffset += 1)
                {
                        int x = warpPatchRectCenter.x - (warpSize.width / 2) + xOffset;
                        int y = warpPatchRectCenter.y - (warpSize.height / 2) + yOffset;
                        cv::Rect testPatchRect(x, y, warpSize.width, warpSize.height);
                        if ((testPatchRect.tl().x >= 0) && (testPatchRect.tl().y >= 0)
                                && (testPatchRect.br().x < warpFrameRect.width)
                                && (testPatchRect.br().y < warpFrameRect.height))
                        {
                            positivePatches.push_back(cv::Rect(x, y, warpSize.width, warpSize.height));
                        }
                }
            }
            for (size_t i = 0; i < warpFrames.size(); ++i)
            {
                concurrent::blockingMap(positivePatches.begin(), positivePatches.end(),
                                        std::bind(&Classifier::train, this, warpFrames.at(i), std::placeholders::_1, 1));
            }
        }
    }
}


cv::Mat Classifier::transform(const cv::Mat& frame, const cv::Point2f& center, const double angle) const
{
    cv::Mat transformMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat transformedFrame;
    cv::warpAffine(frame, transformedFrame, transformMatrix, frame.size());
    cv::Mat integralFrame;
    cv::integral(transformedFrame, integralFrame);
    return integralFrame;
}


void Classifier::init(const cv::Mat& frame, const cv::Rect& patchRect)
{
    for (size_t fern = 0; fern < ferns.size(); ++fern)
    {
        ferns.at(fern)->reset();
    }
    cv::Mat integralFrame;
    cv::integral(frame, integralFrame);
    train(integralFrame, patchRect, 1);
    trainPositive(frame, patchRect);
    trainNegative(integralFrame, patchRect);
}


double Classifier::classify(const cv::Mat& frame, const cv::Rect& patchRect) const
{
    double sum = 0.0;
    for (uint fern = 0; fern < ferns.size(); ++fern)
    {
        sum += ferns.at(fern)->classify(frame, patchRect);
    }
    return (sum / ferns.size());
}


double Classifier::getRectsOverlap(const cv::Rect& first, const cv::Rect& second) const
{
    double overlap = 0.0;
    cv::Rect overlapRect = first & second;
    if (overlapRect.area() > 0)
    {
        overlap = static_cast<double>(overlapRect.area()) / (first.area() + second.area() - overlapRect.area());
    }
//    std::cout << "Overlap = " << overlap << std::endl;
    return overlap;
}


cv::Point2f Classifier::getRectCenter(const cv::Rect& rect) const
{
    float x = static_cast<float>(round(rect.x + (rect.width / 2.0)));
    float y = static_cast<float>(round(rect.y + (rect.height / 2.0)));
    return cv::Point2f(x, y);
}
