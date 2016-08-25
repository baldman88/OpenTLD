#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <algorithm>
#include <vector>
#include <cmath>
#include <memory>
#include <functional>
#include <random>
#include <chrono>
#include <set>

#include <opencv2/imgproc/imgproc.hpp>

#include "Classifier.hpp"
#include "Patch.hpp"
#include "Concurrent.hpp"
#include "KalmanFilter.hpp"


class Detector
{
public:
    explicit Detector(std::shared_ptr<Classifier> &classifier);
    ~Detector() = default;
    void detect(const cv::Mat &frame, const cv::Rect &patchRect, std::vector<Patch> &patches);
    void init(const cv::Mat &frame, const cv::Rect &patchRect);
    void setVarianceThreshold(const cv::Mat &frame, const cv::Rect &patchRect);

private:
    std::shared_ptr<Classifier> classifier;
    int patchRectWidth;
    int patchRectHeight;
    double varianceThreshold;
    int frameWidth;
    int frameHeight;
    const int minSideSize;
    const int maxSideSize;
    cv::Rect lastPatchRect;
    cv::Rect predictedPatchRect;
    KalmanFilter filter;
    int failureCounter;
    int failureScaleFactor;
    cv::Point currentPatchRectCenter;
    cv::Point predictedPatchRectCenter;

    Patch getPatch(const cv::Rect &testRect, const cv::Mat &frame, const cv::Rect &patchRect) const;
    bool checkPatchConformity(const Patch &patch) const;
    double getPatchVariance(const cv::Mat &integralFrame, const cv::Mat &squareIntegralFrame, const cv::Rect &patchRect) const;
    bool checkPatchVariace(const cv::Mat &integralFrame, const cv::Mat &squareIntegralFrame, const cv::Rect &patchRect) const;
    cv::Rect getCurrentPatchRect(const cv::Rect &patchRect);
};

#endif /* DETECTOR_HPP */
