#include "Tracker.hpp"


Tracker::Tracker(std::shared_ptr<Classifier>& classifier)
    : pyramidLevel(5), classifier(classifier)
{
    windowSize = cv::Size(4, 4);
    termCriteria = cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.03);
}


void Tracker::init(const cv::Mat& frame)
{
    prevFrame = frame.clone();
    cv::buildOpticalFlowPyramid(prevFrame, prevFramePyr, windowSize, pyramidLevel, true);
}


Patch Tracker::track(const cv::Mat& frame, const cv::Rect& patchRect)
{
    nextFrame = frame.clone();
    cv::buildOpticalFlowPyramid(nextFrame, nextFramePyr, windowSize, pyramidLevel, true);
    std::vector<cv::Point2f> prevPoints;
    prevPoints = getGridPoints(patchRect);
    std::vector<cv::Point2f> nextPoints(prevPoints);
    std::vector<cv::Point2f> testPoints(prevPoints);
    std::vector<uchar> statusForward;
    std::vector<uchar> statusBackward;
    std::vector<float> errorsForward;
    std::vector<float> errorsBackward;
    cv::calcOpticalFlowPyrLK(prevFramePyr, nextFramePyr, prevPoints, nextPoints, statusForward, errorsForward,
                             windowSize, pyramidLevel, termCriteria, cv::OPTFLOW_USE_INITIAL_FLOW);
    cv::calcOpticalFlowPyrLK(nextFramePyr, prevFramePyr, nextPoints, testPoints, statusBackward, errorsBackward,
                             windowSize, pyramidLevel, termCriteria, cv::OPTFLOW_USE_INITIAL_FLOW);
    std::vector<uchar> status;
    for (uint i = 0; i < statusForward.size(); i++)
    {
        if ((statusForward.at(i) == 1) && (statusBackward.at(i) == 1)
            && (errorsForward.at(i) < 3.0f) && (errorsBackward.at(i) < 3.0f))
        {
            status.push_back(1);
        }
        else
        {
            status.push_back(0);
        }
    }
    std::vector<cv::Point2f> currPrevPoints;
    std::vector<cv::Point2f> currNextPoints;
    std::vector<cv::Point2f> currTestPoints;
    for (uint i = 0; i < status.size(); i++)
    {
        if (status.at(i) == 1)
        {
            currPrevPoints.push_back(prevPoints.at(i));
            currNextPoints.push_back(nextPoints.at(i));
            currTestPoints.push_back(testPoints.at(i));
        }
    }
    std::vector<float> match = getNormCrossCorrelation(currPrevPoints, currNextPoints);
    std::vector<float> confidence = getEuclideanDistance(currPrevPoints, currTestPoints);
    float matchMedian = getMedian(match);
    float confidenceMedian = getMedian(confidence);
    std::vector<cv::Point2f> resultPrevPoints;
    std::vector<cv::Point2f> resultNextPoints;
    for (uint i = 0; i < match.size(); i++)
    {
        if ((match.at(i) >= matchMedian) && (confidence.at(i) <= confidenceMedian))
        {
            resultPrevPoints.push_back(currPrevPoints.at(i));
            resultNextPoints.push_back(currNextPoints.at(i));
        }
    }
    prevFrame = frame.clone();
    prevFramePyr.swap(nextFramePyr);
    Patch trackedPatch;
    if (resultPrevPoints.size() > 0)
    {
        trackedPatch.rect = getBoundedRect(patchRect, resultPrevPoints, resultNextPoints);
    }
    cv::Mat integralFrame;
    cv::integral(frame, integralFrame);
    trackedPatch.confidence = classifier->classify(integralFrame, trackedPatch.rect);
    if ((classifier->getRectsOverlap(patchRect, trackedPatch.rect) > classifier->minOverlap)
        && ((trackedPatch.rect.tl().x >= 0) && (trackedPatch.rect.tl().y >= 0)
        && (trackedPatch.rect.br().x < frame.cols) && (trackedPatch.rect.br().y < frame.rows)))
    {
        trackedPatch.isOverlaps = true;
    }
    return trackedPatch;
}


float Tracker::getMedian(const std::vector<float>& array) const
{
    float median;
    std::vector<float> tmp(array);
    if(tmp.size() == 0)
    {
        median = 0.0f;
    }
    else if (tmp.size() == 1)
    {
        median = tmp.at(0);
    }
    else if (tmp.size() == 2)
    {
        median = ((tmp.at(0) + tmp.at(1)) / 2.0f);
    }
    else
    {
        int index = round(tmp.size() / 2);
        std::sort(tmp.begin(), tmp.end());
        if ((tmp.size() % 2) == 0)
        {
            median = ((tmp.at(index) + tmp.at(index + 1)) / 2.0f);
        }
        else
        {
            median = tmp.at(index);
        }
    }
    return median;
}


std::vector<float> Tracker::getEuclideanDistance(const std::vector<cv::Point2f>& forwardPoints,
                                                 const std::vector<cv::Point2f>& backwardPoints) const
{
    std::vector<float> confidence;
    for (uint i = 0; i < forwardPoints.size(); i++)
    {
        float diffX = pow((forwardPoints.at(i).x - backwardPoints.at(i).x), 2);
        float diffY = pow((forwardPoints.at(i).y - backwardPoints.at(i).y), 2);
        float fbError = sqrt(diffX + diffY);
        confidence.push_back(fbError);
    }
    return confidence;
}


std::vector<float> Tracker::getNormCrossCorrelation(const std::vector<cv::Point2f>& prevPoints,
                                                    const std::vector<cv::Point2f>& nextPoints) const
{
    cv::Mat prevPatch;
    cv::Mat nextPatch;
    cv::Mat result;
    std::vector<float> match;
    for (uint i = 0; i < nextPoints.size(); i++)
    {
        cv::getRectSubPix(prevFrame, cv::Size(PATCH_SIZE, PATCH_SIZE), prevPoints[i], prevPatch);
        cv::getRectSubPix(nextFrame, cv::Size(PATCH_SIZE, PATCH_SIZE), nextPoints[i], nextPatch);
        cv::matchTemplate(prevPatch, nextPatch, result, cv::TM_CCOEFF);
        match.push_back(result.at<float>(0, 0));
    }
    return match;
}


std::vector<cv::Point2f> Tracker::getGridPoints(const cv::Rect& rect) const
{
    cv::Rect localRect;
    localRect.x = rect.x + (PATCH_SIZE / 2);
    localRect.y = rect.y + (PATCH_SIZE / 2);
    localRect.width = rect.width - PATCH_SIZE;
    localRect.height = rect.height - PATCH_SIZE;
    float stepByWidth = static_cast<float>(localRect.width) / (DIM_POINTS - 1);
    float stepByHeight = static_cast<float>(localRect.height) / (DIM_POINTS - 1);
    std::vector<cv::Point2f> gridPoints;
    for (uint i = 0; i < DIM_POINTS; i++)
    {
        for (uint j = 0; j < DIM_POINTS; j++)
        {
            float x = localRect.x + (stepByWidth * i);
            float y = localRect.y + (stepByHeight * j);
            gridPoints.push_back(cv::Point2f(x, y));
        }
    }
    return gridPoints;
}


cv::Rect Tracker::getBoundedRect(const cv::Rect& rect, const std::vector<cv::Point2f>& prevPoints,
                                 const std::vector<cv::Point2f>& nextPoints) const
{
    std::vector<float> diffX;
    std::vector<float> diffY;
    for (uint point = 0; point < prevPoints.size(); point++)
    {
        diffX.push_back(nextPoints.at(point).x - prevPoints.at(point).x);
        diffY.push_back(nextPoints.at(point).y - prevPoints.at(point).y);
    }
    float dX = getMedian(diffX);
    float dY = getMedian(diffY);
    std::vector<float> pointsShift;
    for (uint i = 0; i < (prevPoints.size() - 1); i++)
    {
        for (uint j = (i + 1); j < prevPoints.size(); j++)
        {
            float distPrev = sqrt(pow((prevPoints.at(i).x - prevPoints.at(j).x), 2) + pow((prevPoints.at(i).y - prevPoints.at(j).y), 2));
            float distNext = sqrt(pow((nextPoints.at(i).x - nextPoints.at(j).x), 2) + pow((nextPoints.at(i).y - nextPoints.at(j).y), 2));
            pointsShift.push_back(distNext / distPrev);
        }
    }
    float shift = getMedian(pointsShift);
    float shiftW = 0.5f * (shift - 1) * rect.width;
    float shiftH = 0.5f * (shift - 1) * rect.height;
    cv::Rect boundedRect;
    boundedRect.x = rect.x - round(shiftW - dX);
    boundedRect.y = rect.y - round(shiftH - dY);
    boundedRect.width = rect.width + round(shiftW * 2);
    boundedRect.height = rect.height + round(shiftH * 2);
    return boundedRect;
}
