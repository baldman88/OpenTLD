#include "Feature.hpp"

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
}

int Feature::test(const cv::Mat& frame, const cv::Rect& patch)
{
    int x = round(scaleX * patch.width) + patch.x;
    int y = round(scaleY * patch.height) + patch.y;
    int w = round(scaleW * patch.width);
    int h = round(scaleH * patch.height);
    int left = sumRect(frame, cv::Rect(x, y, (w / 2), h));
    int right = sumRect(frame, cv::Rect((x + (w / 2)), y, (w / 2), h));
    int top = sumRect(frame, cv::Rect(x, y, w, (h / 2)));
    int bottom = sumRect(frame, cv::Rect(x, (y + (h / 2)), w, (h / 2)));
    int result = ((left > right) ? 0 : 2) + ((top > bottom) ? 0 : 1);
    return result;
}

int Feature::sumRect(const cv::Mat& frame, const cv::Rect& rect)
{
    int summa = frame.at<int>(cv::Point(rect.x + rect.width, rect.y + rect.height))
                + frame.at<int>(cv::Point(rect.x, rect.y))
                - frame.at<int>(cv::Point(rect.x + rect.width, rect.y))
                - frame.at<int>(cv::Point(rect.x, rect.y + rect.height));
    return summa;
}
