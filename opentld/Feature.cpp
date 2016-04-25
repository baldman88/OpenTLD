#include "Feature.hpp"
#include <iostream>


Feature::Feature(const double minScale)
{
    std::random_device randomDevice;
    std::mt19937 randomEngine(randomDevice());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    /* scaleW and scaleH in range minScale .. 1.0 */
    scaleW = ((1.0 - minScale) * distribution(randomEngine)) + minScale;
    scaleH = ((1.0 - minScale) * distribution(randomEngine)) + minScale;
    /* scaleX and scaleY in range 0.0 .. (1.0 - minScale) */
    scaleX = (1.0 - scaleW) * distribution(randomEngine);
    scaleY = (1.0 - scaleH) * distribution(randomEngine);
//    std::cout << "sX = " << scaleX << "; sY = " << scaleY << "; sW = " << scaleW << "; sH = " << scaleH << std::endl;
}


int Feature::test(const cv::Mat& frame, const cv::Rect& patchRect)
{
    int x = static_cast<int>(round(scaleX * patchRect.width)) + patchRect.x;
    int y = static_cast<int>(round(scaleY * patchRect.height)) + patchRect.y;
    int w = static_cast<int>(round(scaleW * patchRect.width));
    int h = static_cast<int>(round(scaleH * patchRect.height));
    int left = sumRect(frame, cv::Rect(x, y, (w / 2), h));
    int right = sumRect(frame, cv::Rect((x + (w / 2)), y, (w / 2), h));
    int top = sumRect(frame, cv::Rect(x, y, w, (h / 2)));
    int bottom = sumRect(frame, cv::Rect(x, (y + (h / 2)), w, (h / 2)));
    return ((left > right) ? 0 : 2) + ((top > bottom) ? 0 : 1);
}


int Feature::sumRect(const cv::Mat& frame, const cv::Rect& patchRect)
{
    return (frame.at<int>(cv::Point(patchRect.x + patchRect.width, patchRect.y + patchRect.height))
            + frame.at<int>(cv::Point(patchRect.x, patchRect.y))
            - frame.at<int>(cv::Point(patchRect.x + patchRect.width, patchRect.y))
            - frame.at<int>(cv::Point(patchRect.x, patchRect.y + patchRect.height)));
}
