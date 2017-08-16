#include "Classifier.hpp"
#include <iostream>


Classifier::Classifier(const int fernsCount, const int featuresCount, const double minFeatureScale, const double maxFeatureScale)
{
    for (int fern = 0; fern < fernsCount; ++fern)
    {
        ferns.push_back(std::make_shared<Fern>(featuresCount, minFeatureScale, maxFeatureScale));
    }
}


void Classifier::train(const cv::Mat &frame, const cv::Rect &patchRect, const bool isPositive)
{
    for (auto fern: ferns)
    {
        fern->train(frame, patchRect, isPositive);
    }
}


void Classifier::trainNegative(const cv::Mat &frame, const cv::Rect &patchRect)
{
    double minScale = 0.5;
    double maxScale = 1.5;
    double scaleStep = 0.25;
    for (double scale = minScale; scale <= maxScale; scale += scaleStep)
    {
        int xMin = 0;
        int currentWidth = static_cast<int>(round(scale * patchRect.width));
        int xMax = frame.cols - currentWidth;
        int xStep = 10;
        for (int x = xMin; x < xMax; x += xStep)
        {
            int yMin = 0;
            int currentHeight = static_cast<int>(round(scale * patchRect.height));
            int yMax = frame.rows - currentHeight;
            int yStep = 10;
            for (int y = yMin; y < yMax; y += yStep)
            {
                cv::Rect negativePatchRect(x, y, currentWidth, currentHeight);
                if (getRectsOverlap(patchRect, negativePatchRect) < minimumOverlap)
                {
                    train(frame, negativePatchRect, false);
                }
            }
        }
    }
}


void Classifier::trainPositive(const cv::Mat &frame, const cv::Rect &patchRect)
{
    cv::Point2f patchRectCenter = getRectCenter(patchRect);

    std::set<int> widths;
    std::set<int> heights;

    for (double scale = 0.95; scale <= 1.05; scale += 0.05)
    {
        int width = static_cast<int>(round(patchRect.width * scale));
        int minX = (patchRectCenter.x - static_cast<int>(round(width / 2))) - 3;
        int maxX = (patchRectCenter.x + static_cast<int>(round(width / 2))) + 3;
        if ((minX >= 0) && (maxX < frame.cols))
        {
            widths.insert(width);
        }

        int height = static_cast<int>(round(patchRect.height * scale));
        int minY = (patchRectCenter.y - static_cast<int>(round(height / 2))) - 3;
        int maxY = (patchRectCenter.y + static_cast<int>(round(height / 2))) + 3;
        if ((minY >= 0) && (maxY < frame.rows))
        {
            heights.insert(height);
        }
    }

    if ((widths.empty() != true) && (heights.empty() != true))
    {
        cv::Size warpFrameSize(*(widths.rbegin()), *(heights.rbegin()));
        double maxAngle;
        cv::Rect warpFrameRect;
        for (maxAngle = 10.0; maxAngle >= 0.0; maxAngle -= 1.0)
        {
            warpFrameRect = cv::RotatedRect(patchRectCenter, warpFrameSize, maxAngle).boundingRect();
            if ((warpFrameRect.tl().x >= 0)
                && (warpFrameRect.tl().y >= 0)
                && (warpFrameRect.br().x < frame.cols)
                && (warpFrameRect.br().y < frame.rows))
            {
                break;
            }
        }

        cv::Rect warpPatchRect((patchRect.x - warpFrameRect.x), (patchRect.y - warpFrameRect.y), patchRect.width, patchRect.height);
        cv::Mat warpFrame = frame(warpFrameRect);
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


        for (auto widthsIter = widths.begin(); widthsIter != widths.end(); ++widthsIter)
        {
            int width = *widthsIter;
            for (auto heightsIter = heights.begin(); heightsIter != heights.end(); ++heightsIter)
            {
                int height = *heightsIter;
                for (int xOffset = -1; xOffset <= 1; xOffset += 1)
                {
                    for (int yOffset = -1; yOffset <= 1; yOffset += 1)
                    {
                        int x = warpPatchRectCenter.x - static_cast<int>(round(width / 2)) + xOffset;
                        int y = warpPatchRectCenter.y - static_cast<int>(round(height / 2)) + yOffset;
                        cv::Rect testPatchRect(x, y, width, height);
                        if ((testPatchRect.tl().x >= 0)
                            && (testPatchRect.tl().y >= 0)
                            && (testPatchRect.br().x < warpFrameRect.width)
                            && (testPatchRect.br().y < warpFrameRect.height))
                        {
                            positivePatches.push_back(cv::Rect(x, y, width, height));
                        }
                    }
                }
            }
        }
        for (size_t i = 0; i < warpFrames.size(); ++i)
        {
            concurrent::blockingMap(positivePatches.begin(), positivePatches.end(),
                                    std::bind(&Classifier::train, this, warpFrames.at(i), std::placeholders::_1, true));
        }
    }
}


cv::Mat Classifier::transform(const cv::Mat &frame, const cv::Point2f &center, const double angle) const
{
    cv::Mat transformMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat transformedFrame;
    cv::warpAffine(frame, transformedFrame, transformMatrix, frame.size());
    cv::Mat integralFrame;
    cv::integral(transformedFrame, integralFrame);
    return integralFrame;
}


void Classifier::init(const cv::Mat &frame, const cv::Rect &patchRect)
{
    for (size_t fern = 0; fern < ferns.size(); ++fern)
    {
        ferns.at(fern)->reset();
    }
    cv::Mat integralFrame;
    cv::integral(frame, integralFrame);
    train(integralFrame, patchRect, true);
    trainPositive(frame, patchRect);
    trainNegative(integralFrame, patchRect);
}


double Classifier::classify(const cv::Mat &frame, const cv::Rect &patchRect) const
{
    double sum = 0.0;
    for (uint fern = 0; fern < ferns.size(); ++fern)
    {
        sum += ferns.at(fern)->classify(frame, patchRect);
    }
    return (sum / ferns.size());
}


double Classifier::getRectsOverlap(const cv::Rect &first, const cv::Rect &second) const
{
    double overlap = 0.0;
    cv::Rect overlapRect = first & second;
    if (overlapRect.area() > 0)
    {
        overlap = static_cast<double>(overlapRect.area()) / (first.area() + second.area() - overlapRect.area());
    }
    return overlap;
}


cv::Point2f Classifier::getRectCenter(const cv::Rect &rect) const
{
    float x = static_cast<float>(round(rect.x + (rect.width / 2.0)));
    float y = static_cast<float>(round(rect.y + (rect.height / 2.0)));
    return cv::Point2f(x, y);
}
