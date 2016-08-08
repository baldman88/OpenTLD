#include "KalmanFilter.hpp"


KalmanFilter::KalmanFilter()
{
    init();
}


cv::Rect KalmanFilter::predict(const cv::Rect &rect)
{
    cv::Rect result(0, 0, 0, 0);

    previousTicks = currentTicks;
    currentTicks = cv::getTickCount();
    float dT = static_cast<float>(currentTicks - previousTicks) / cv::getTickFrequency();

    if (lostCounter < 50) {
        filter.transitionMatrix.at<float>(2) = dT;
        filter.transitionMatrix.at<float>(9) = dT;
        state = filter.predict();

        result.width = state.at<float>(4);
        result.height = state.at<float>(5);
        result.x = state.at<float>(0) - (result.width / 2);
        result.y = state.at<float>(1) - (result.height / 2);
    }

    if (rect.area() > 0) {
        measurment.at<float>(0) = rect.x + (rect.width / 2);
        measurment.at<float>(1) = rect.y + (rect.height / 2);
        measurment.at<float>(2) = static_cast<float>(rect.width);
        measurment.at<float>(3) = static_cast<float>(rect.height);

        if (isInitialized == false) {
            filter.errorCovPre.at<float>(0) = 1;
            filter.errorCovPre.at<float>(7) = 1;
            filter.errorCovPre.at<float>(14) = 1;
            filter.errorCovPre.at<float>(21) = 1;
            filter.errorCovPre.at<float>(28) = 1;
            filter.errorCovPre.at<float>(35) = 1;

            state.at<float>(0) = measurment.at<float>(0);
            state.at<float>(1) = measurment.at<float>(1);
            state.at<float>(2) = 0;
            state.at<float>(3) = 0;
            state.at<float>(4) = measurment.at<float>(2);
            state.at<float>(5) = measurment.at<float>(3);

            isInitialized = true;
        } else {
            filter.correct(measurment);
        }

        lostCounter = 0;
    } else {
        lostCounter++;
    }

    return result;
}


void KalmanFilter::reset()
{
    init();
}


void KalmanFilter::init()
{
    filter = cv::KalmanFilter(6, 4, 0, CV_32F);

    state = cv::Mat(6, 1, CV_32F);
    measurment = cv::Mat(4, 1, CV_32F);

    cv::setIdentity(filter.transitionMatrix);

    filter.measurementMatrix = cv::Mat::zeros(4, 6, CV_32F);
    filter.measurementMatrix.at<float>(0) = 1.0f;
    filter.measurementMatrix.at<float>(7) = 1.0f;
    filter.measurementMatrix.at<float>(16) = 1.0f;
    filter.measurementMatrix.at<float>(23) = 1.0f;

    filter.processNoiseCov.at<float>(0) = 1e-2;
    filter.processNoiseCov.at<float>(7) = 1e-2;
    filter.processNoiseCov.at<float>(14) = 5.0f;
    filter.processNoiseCov.at<float>(21) = 5.0f;
    filter.processNoiseCov.at<float>(28) = 1e-2;
    filter.processNoiseCov.at<float>(35) = 1e-2;

    cv::setIdentity(filter.measurementNoiseCov, cv::Scalar(1e-1));

    currentTicks = 0;
    lostCounter = 0;
    isInitialized = false;
}
