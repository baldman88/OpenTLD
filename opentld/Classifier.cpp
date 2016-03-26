#include "Classifier.hpp"


Classifier::Classifier(const int fernsCount, const int featuresCount, const double minFeatureScale)
{
    for (int fern = 0; fern < fernsCount; ++fern)
    {
        ferns.push_back(std::make_shared<Fern>(featuresCount, minFeatureScale));
    }
}


void Classifier::train(const cv::Mat& frame, const cv::Rect& patch, const int patchClass)
{
    for (auto fern: ferns)
    {
        fern->train(frame, patch, patchClass);
    }
}


void Classifier::trainNegative(const cv::Mat& frame, const cv::Rect& patch)
{
    double minScale = 0.5;
    double maxScale = 1.5;
    double scaleStep = 0.25;
    for (double scale = minScale; scale <= maxScale; scale += scaleStep)
    {
        int xMin = 0;
        int currentWidth = static_cast<int>(scale * patch.width);
        int xMax = frame.cols - currentWidth;
        int xStep = 10;
        for (int x = xMin; x < xMax; x += xStep)
        {
            int yMin = 0;
            int currentHeight = static_cast<int>(scale * patch.height);
            int yMax = frame.rows - currentHeight;
            int yStep = 10;
            for (int y = yMin; y < yMax; y += yStep)
            {
                cv::Rect negativePatch(x, y, currentWidth, currentHeight);
                if (getPatchesOverlap(patch, negativePatch) < 0.6)
                {
                    train(frame, negativePatch, 0);
                }
            }
        }
    }
}


void Classifier::trainPositive(const cv::Mat& frame, const cv::Rect& patch)
{
    cv::Point2f patchCenter = getPatchCenter(patch);
    double maxAngle;
    cv::Rect warpRect;
    /* warpSize takes into account the size of the maximum values of the offset (±3 pixels) and scaling (1.1) */
    cv::Size warpSize(static_cast<int>(std::ceil(patch.width * 1.1 + 6)),
                      static_cast<int>(std::ceil(patch.height * 1.1 + 6)));
    for (maxAngle = 10.0; maxAngle > 0.0; maxAngle -= 1.0)
    {
        warpRect = cv::RotatedRect(patchCenter, warpSize, maxAngle).boundingRect();
        if ((warpRect.tl().x >= 0) && (warpRect.tl().y >= 0)
            && (warpRect.br().x < frame.cols) && (warpRect.br().y < frame.rows))
        {
            break;
        }
    }
    cv::Mat warpFrame = frame(warpRect);
    cv::Rect warpPatch((patch.x - warpRect.x), (patch.y - warpRect.y), patch.width, patch.height);
    std::vector<double> angles;
    for (double angle = -maxAngle; angle <= maxAngle; angle += 1.0)
    {
        angles.push_back(angle);
    }
    cv::Point2f warpPatchCenter = getPatchCenter(warpPatch);
    std::vector<cv::Mat> warpFrames(angles.size());
    concurrent::blockingMapped(angles.begin(), angles.end(), warpFrames.begin(),
                               std::bind(&Classifier::transform, this, warpFrame,
                                         warpPatchCenter, std::placeholders::_1));
    std::vector<cv::Rect> positivePatches;
    for (int xOffset = -3; xOffset <= 3; xOffset += 1)
    {
        for (int yOffset = -3; yOffset <= 3; yOffset += 1)
        {
            for (double scale = 0.9; scale <= 1.1; scale += 0.1)
            {
                int width = round(warpPatch.width * scale);
                int height = round(warpPatch.height * scale);
                int x = warpPatchCenter.x - (width / 2) + xOffset;
                int y = warpPatchCenter.y - (height / 2) + yOffset;
                positivePatches.push_back(cv::Rect(x, y, width, height));
            }
        }
    }
    for (size_t i = 0; i < warpFrames.size(); ++i)
    {
        concurrent::blockingMap(positivePatches.begin(), positivePatches.end(),
                                std::bind(&Classifier::train, this, warpFrames.at(i),
                                          std::placeholders::_1, 1));
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


void Classifier::init(const cv::Mat& frame, const cv::Rect& patch)
{
    for (size_t fern = 0; fern < ferns.size(); ++fern)
    {
        ferns.at(fern)->reset();
    }
    cv::Mat integralFrame;
    cv::integral(frame, integralFrame);
    train(integralFrame, patch, 1);
    trainPositive(frame, patch);
    trainNegative(integralFrame, patch);
}


double Classifier::classify(const cv::Mat& frame, const cv::Rect& patch) const
{
    double sum = 0.0;
    for (uint fern = 0; fern < ferns.size(); ++fern)
    {
        sum += ferns.at(fern)->classify(frame, patch);
    }
    return (sum / ferns.size());
}


double Classifier::getPatchesOverlap(const cv::Rect& patch1, const cv::Rect& patch2) const
{
    double overlap = 0.0;
    cv::Rect overlapRect = patch1 & patch2;
    if (overlapRect.area() > 0)
    {
        overlap = static_cast<double>(overlapRect.area()) / (patch1.area() + patch2.area() - overlapRect.area());
    }
    return overlap;
}


cv::Point2f Classifier::getPatchCenter(const cv::Rect& rect) const
{
    float x = static_cast<float>(round(rect.x + (rect.width / 2.0)));
    float y = static_cast<float>(round(rect.y + (rect.height / 2.0)));
    return cv::Point2f(x, y);
}
