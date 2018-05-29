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
    calculateScaledSizes(patchRect, frame.size(), widths, heights);

    if ((widths.empty() != true) && (heights.empty() != true))
    {
        double maxAngle = getMaxRotateAngle(patchRect, frame.size(), widths, heights);
        cv::Size warpFrameSize(*(widths.rbegin()), *(heights.rbegin()));
        cv::Rect warpFrameRect = cv::RotatedRect(patchRectCenter, warpFrameSize, maxAngle).boundingRect();

        cv::Rect warpPatchRect((patchRect.x - warpFrameRect.x), (patchRect.y - warpFrameRect.y), patchRect.width, patchRect.height);
        cv::Mat warpFrame = frame(warpFrameRect);
        std::vector<double> angles;
        for (double angle = -maxAngle; angle <= maxAngle; angle += 1.0)
        {
            angles.push_back(angle);
        }

        cv::Point2f warpPatchRectCenter = getRectCenter(warpPatchRect);

#ifdef USE_DEBUG_INFO
        auto start = std::chrono::high_resolution_clock::now();
#endif // USE_LOGGING

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
        std::cout << "Classifier for warpFrames elapsed = "
                << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
        start = stop;
#endif // USE_LOGGING

//        if (flag == false)
//        {
//            std::cout << "Write images ..." << std::endl;
//            cv::Rect target(warpPatchRectCenter.x - static_cast<int>(round(patchRect.width / 2)),
//                            warpPatchRectCenter.y - static_cast<int>(round(patchRect.height / 2)),
//                            patchRect.width, patchRect.height);
//            for (size_t i = 0; i < warpFrames.size(); ++i)
//            {
//                cv::Mat tmp = warpFrames.at(i).clone();
//                cv::rectangle(tmp, target, cv::Scalar(0, 255, 0));
//                cv::imwrite("/home/baldman/1/" + std::to_string(i) + ".png", tmp);
//            }
//            flag = true;
//        }

        tmpFrames = warpFrames;
        concurrent::blockingMapped(tmpFrames.begin(), tmpFrames.end(), warpFrames.begin(),
                                   std::bind(&Classifier::getIntegralFrame, this, std::placeholders::_1));

#ifdef USE_DEBUG_INFO
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Classifier for integralFrames elapsed = "
                << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
        start = stop;
#endif // USE_LOGGING

//        std::vector<cv::Mat> warpFrames(angles.size());
//        concurrent::blockingMapped(angles.begin(), angles.end(), warpFrames.begin(),
//                                   std::bind(&Classifier::transform, this, warpFrame, warpPatchRectCenter, std::placeholders::_1));

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

#ifdef USE_DEBUG_INFO
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Classifier for make positive patches elapsed = "
                << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
        start = stop;
#endif // USE_LOGGING

//        using positivePatchesIterator = typename decltype(positivePatches)::iterator;
        for (size_t i = 0; i < warpFrames.size(); ++i)
        {
//            concurrent::blockingMap(positivePatches.begin(), positivePatches.end(),
//                                    std::bind(&Classifier::train, this, warpFrames.at(i), std::placeholders::_1, true));
            concurrent::blockingMap(positivePatches.begin(), positivePatches.end(),
                                    std::bind(&Classifier::trainOnRange<decltype(positivePatches)::iterator>,
                                              this, warpFrames.at(i), std::placeholders::_1, std::placeholders::_2, true));
//            trainOnRange<decltype(positivePatches)::iterator>(warpFrames.at(i), positivePatches.begin(), positivePatches.end(), true);
        }

#ifdef USE_DEBUG_INFO
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Classifier for training elapsed = "
                << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
#endif // USE_LOGGING

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


void Classifier::calculateScaledSizes(const cv::Rect& patchRect, const cv::Size& frameSize, std::set<int>& widths, std::set<int>& heights) const
{
    cv::Point2f patchRectCenter = getRectCenter(patchRect);
    for (double scale = 0.95; scale <= 1.05; scale += 0.05)
    {
        int width = static_cast<int>(round(patchRect.width * scale));
        int minX = (patchRectCenter.x - static_cast<int>(round(width / 2))) - 3;
        int maxX = (patchRectCenter.x + static_cast<int>(round(width / 2))) + 3;
        if ((minX >= 0) && (maxX < frameSize.width))
        {
            widths.insert(width);
        }

        int height = static_cast<int>(round(patchRect.height * scale));
        int minY = (patchRectCenter.y - static_cast<int>(round(height / 2))) - 3;
        int maxY = (patchRectCenter.y + static_cast<int>(round(height / 2))) + 3;
        if ((minY >= 0) && (maxY < frameSize.height))
        {
            heights.insert(height);
        }
    }
}


double Classifier::getMaxRotateAngle(const cv::Rect& patchRect, const cv::Size& frameSize,
                                     const std::set<int>& widths, const std::set<int>& heights) const
{
    cv::Point2f patchRectCenter = getRectCenter(patchRect);
    cv::Size warpFrameSize(*(widths.rbegin()), *(heights.rbegin()));
    double maxAngle;
    cv::Rect warpFrameRect;
    for (maxAngle = 10.0; maxAngle >= 0.0; maxAngle -= 1.0)
    {
        cv::Rect warpFrameRect = cv::RotatedRect(patchRectCenter, warpFrameSize, maxAngle).boundingRect();
        if ((warpFrameRect.tl().x >= 0)
            && (warpFrameRect.tl().y >= 0)
            && (warpFrameRect.br().x < frameSize.width)
            && (warpFrameRect.br().y < frameSize.height))
        {
            break;
        }
    }
    return maxAngle;
}
