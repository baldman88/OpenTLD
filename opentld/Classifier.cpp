#include "Classifier.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>


Classifier::Classifier(const int fernsCount, const int featuresCount, const double minFeatureScale, const double maxFeatureScale)
{
    for (int fern = 0; fern < fernsCount; ++fern)
    {
        ferns.push_back(std::make_shared<Fern>(featuresCount, minFeatureScale, maxFeatureScale));
    }
}


void Classifier::train(const cv::Mat &frame, const cv::Rect &patchRect, const bool isPositive)
{
    for (const auto& fern: ferns)
    {
        fern->train(frame, patchRect, isPositive);
    }
}


void Classifier::trainNegative(const cv::Mat &frame, const cv::Rect &patchRect)
{
    double minScale = 0.5;
    double maxScale = 1.5;
    double scaleStep = 0.25;
    double scale = minScale;
    while (scale <= maxScale)
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
        scale += scaleStep;
    }
}


void Classifier::trainPositive(const cv::Mat &frame, const cv::Rect &patchRect)
{
    cv::Point2f patchRectCenter = getRectCenter(patchRect);

    std::set<int> widths;
    std::set<int> heights;
    calculateScaledSizes(patchRect, frame.size(), widths, heights);

    if (!widths.empty() && !heights.empty())
    {
        double maxAngle = getMaxRotateAngle(patchRect, frame.size(), widths, heights);
        cv::Size warpFrameSize(*(widths.rbegin()), *(heights.rbegin()));
        cv::Rect warpFrameRect = cv::RotatedRect(patchRectCenter, warpFrameSize, maxAngle).boundingRect();

        cv::Rect warpPatchRect((patchRect.x - warpFrameRect.x), (patchRect.y - warpFrameRect.y), patchRect.width, patchRect.height);
        cv::Mat warpFrame = frame(warpFrameRect);
        std::vector<double> angles;
        double angle = -maxAngle;
        while (angle <= maxAngle)
        {
            angles.push_back(angle);
            angle += 1.0;
        }

        cv::Point2f warpPatchRectCenter = getRectCenter(warpPatchRect);

#ifdef USE_DEBUG_INFO
        auto start = std::chrono::high_resolution_clock::now();
#endif // USE_DEBUG_INFO

        std::vector<cv::Mat> tmpFrames;
        tmpFrames.push_back(warpFrame);
        cv::Mat tmp;
        cv::flip(warpFrame, tmp, 1);
        tmpFrames.push_back(tmp.clone());
        cv::flip(warpFrame, tmp, 0);
        tmpFrames.push_back(tmp.clone());
        cv::flip(warpFrame, tmp, -1);
        tmpFrames.push_back(tmp.clone());

        std::vector<cv::Mat> warpFrames;

        for (size_t i = 0; i < tmpFrames.size(); ++i)
        {
            for (size_t j = 0; j < angles.size(); ++j)
            {
                warpFrames.push_back(transform(tmpFrames.at(i), warpPatchRectCenter, angles.at(j)));
            }
        }

#ifdef USE_DEBUG_INFO
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Classifier: for warpFrames elapsed = "
                << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;
        start = stop;
#endif // USE_DEBUG_INFO

        concurrent::blockingMapped(warpFrames.begin(), warpFrames.end(), warpFrames.begin(),
                                   std::bind(&Classifier::getIntegralFrame, this, std::placeholders::_1));

#ifdef USE_DEBUG_INFO
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Classifier: for integralFrames elapsed = "
                << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;
        start = stop;
#endif // USE_DEBUG_INFO

        std::vector<cv::Rect> positivePatches;
        for (const auto width : widths)
        {
            for (const auto height : heights)
            {
                for (int xOffset = -1; xOffset <= 1; xOffset += 1)
                {
                    for (int yOffset = -1; yOffset <= 1; yOffset += 1)
                    {
                        int x = static_cast<int>(warpPatchRectCenter.x - static_cast<int>(round(width / 2)) + xOffset);
                        int y = static_cast<int>(warpPatchRectCenter.y - static_cast<int>(round(height / 2)) + yOffset);
                        cv::Rect testPatchRect(x, y, width, height);
                        if ((testPatchRect.tl().x >= 0)
                            && (testPatchRect.tl().y >= 0)
                            && (testPatchRect.br().x < warpFrameRect.width)
                            && (testPatchRect.br().y < warpFrameRect.height))
                        {
                            positivePatches.emplace_back(cv::Rect(x, y, width, height));
                        }
                    }
                }
            }
        }

#ifdef USE_DEBUG_INFO
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Classifier: for make positive patches elapsed = "
                << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;
        start = stop;
#endif // USE_DEBUG_INFO

        std::cout << "Classifier: warpFrames.size() = " << warpFrames.size() << std::endl;
        for (const auto frame : warpFrames)
        {
            concurrent::blockingMap(positivePatches.begin(), positivePatches.end(),
                                    std::bind(&Classifier::train, this, frame, std::placeholders::_1, true));
        }

#ifdef USE_DEBUG_INFO
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Classifier: for training elapsed = "
                << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms" << std::endl;
#endif // USE_DEBUG_INFO

    }
}


cv::Mat Classifier::transform(const cv::Mat &frame, const cv::Point2f &center, const double angle) const
{
    cv::Mat transformMatrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat transformedFrame;
    cv::warpAffine(frame, transformedFrame, transformMatrix, frame.size());
    return transformedFrame;
}


cv::Mat Classifier::flipVertical(const cv::Mat &frame) const
{
    cv::Mat flippedFrame;
    cv::flip(frame, flippedFrame, 0);
    return flippedFrame;
}


cv::Mat Classifier::flipHorizontal(const cv::Mat &frame) const
{
    cv::Mat flippedFrame;
    cv::flip(frame, flippedFrame, 1);
    return flippedFrame;
}


cv::Mat Classifier::getIntegralFrame(const cv::Mat &frame) const
{
    cv::Mat integralFrame;
    cv::integral(frame, integralFrame);
    return integralFrame;
}


void Classifier::init(const cv::Mat &frame, const cv::Rect &patchRect)
{
    for (const auto& fern : ferns)
    {
        fern->reset();
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
    for (const auto& fern : ferns)
    {
        sum += fern->classify(frame, patchRect);
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
    auto x = static_cast<float>(round(rect.x + (rect.width / 2.0)));
    auto y = static_cast<float>(round(rect.y + (rect.height / 2.0)));
    return cv::Point2f(x, y);
}


void Classifier::calculateScaledSizes(const cv::Rect& patchRect, const cv::Size& frameSize, std::set<int>& widths, std::set<int>& heights) const
{
    cv::Point2f patchRectCenter = getRectCenter(patchRect);
    double scale = 0.95;
    while (scale <= 1.05)
    {
        int width = static_cast<int>(round(patchRect.width * scale));
        int minX = static_cast<int>((patchRectCenter.x - static_cast<int>(round(width / 2)))) - 3;
        int maxX = static_cast<int>((patchRectCenter.x + static_cast<int>(round(width / 2)))) + 3;
        if ((minX >= 0) && (maxX < frameSize.width))
        {
            widths.insert(width);
        }

        int height = static_cast<int>(round(patchRect.height * scale));
        int minY = static_cast<int>((patchRectCenter.y - static_cast<int>(round(height / 2)))) - 3;
        int maxY = static_cast<int>((patchRectCenter.y + static_cast<int>(round(height / 2)))) + 3;
        if ((minY >= 0) && (maxY < frameSize.height))
        {
            heights.insert(height);
        }
        scale += 0.05;
    }
}


double Classifier::getMaxRotateAngle(const cv::Rect& patchRect, const cv::Size& frameSize,
                                     const std::set<int>& widths, const std::set<int>& heights) const
{
    cv::Point2f patchRectCenter = getRectCenter(patchRect);
    cv::Size warpFrameSize(*(widths.rbegin()), *(heights.rbegin()));
    float maxAngle = 10.0;
    while (maxAngle >= 0.0)
    {
        cv::Rect boundingRect = cv::RotatedRect(patchRectCenter, warpFrameSize, maxAngle).boundingRect();
        maxAngle -= 1.0;
        if ((boundingRect.tl().x >= 0)
            && (boundingRect.tl().y >= 0)
            && (boundingRect.br().x < frameSize.width)
            && (boundingRect.br().y < frameSize.height))
        {
            break;
        }
    }
    return maxAngle;
}
