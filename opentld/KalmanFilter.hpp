#ifndef KALMANFILTER_HPP
#define KALMANFILTER_HPP

#include <opencv2/opencv.hpp>

#include <iostream>


class KalmanFilter
{
public:
    KalmanFilter();
    cv::Rect predict(const cv::Rect &rect);
    void reset();

private:
    cv::KalmanFilter filter;
    cv::Mat state;
    cv::Mat measurment;
    int64_t previousTicks;
    int64_t currentTicks;
    int lostCounter;
    bool isInitialized;
    void init();
};


#endif /* KALMANFILTER_HPP */
