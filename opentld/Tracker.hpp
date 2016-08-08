#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <vector>
#include <memory>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "Classifier.hpp"
#include "Patch.hpp"


#define DIM_POINTS 20

#define TOTAL_POINTS (DIM_POINTS * DIM_POINTS)

#define PATCH_SIZE 10


class Tracker
{
public:
    explicit Tracker(std::shared_ptr<Classifier> &classifier);
    ~Tracker() = default;
    void init(const cv::Mat &frame);
    Patch track(const cv::Mat &frame, const cv::Rect &patchRect);

private:
    const int pyramidLevel;
    cv::Mat prevFrame;
    cv::Mat nextFrame;
    std::vector<cv::Mat> prevFramePyr;
    std::vector<cv::Mat> nextFramePyr;
    cv::Size windowSize;
    cv::TermCriteria termCriteria;
    std::shared_ptr<Classifier> classifier;
    int templateSize;

    double getMedian(const std::vector<double> &array) const;
    std::vector<double> getEuclideanDistance(const std::vector<cv::Point2f> &forwardPoints,
                                            const std::vector<cv::Point2f> &backwardPoints) const;
    std::vector<double> getNormCrossCorrelation(const std::vector<cv::Point2f> &prevPoints,
                                               const std::vector<cv::Point2f> &nextPoints) const;
    std::vector<cv::Point2f> getGridPoints(const cv::Rect &rect) const;
    cv::Rect getBoundedRect(const cv::Rect &rect, const std::vector<cv::Point2f> &prevPoints,
                            const std::vector<cv::Point2f> &nextPoints) const;
};

#endif /* TRACKER_HPP */

