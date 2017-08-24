#include <iostream>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "opentld/TLDTracker.hpp"

#define DEBUG_INFO 1

TLDTracker tracker;
cv::Rect roi;
bool isSelectionActive = false;
bool isTargetSelected = false;


void mouseHandler(int event, int x, int y, int flags, void* param)
{
    if (event == CV_EVENT_RBUTTONDOWN)
    {
        tracker.resetTracker();
        roi = cv::Rect(0, 0, 0, 0);
        isTargetSelected = false;
    }
    if ((event == CV_EVENT_LBUTTONDOWN) && (isTargetSelected == false))
    {
        isSelectionActive = true;
        roi.x = x;
        roi.y = y;
    }
    if ((event == CV_EVENT_MOUSEMOVE) && (isSelectionActive == true))
    {
        roi.width = x - roi.x;
        roi.height = y - roi.y;
    }
    if (event == CV_EVENT_LBUTTONUP)
    {
        isSelectionActive = false;
        roi.width = x - roi.x;
        roi.height = y - roi.y;
        if ((roi.width < 16) || (roi.height < 16) /*|| (roi.width > 120) || (roi.height > 120)*/)
        {
            roi = cv::Rect(0, 0, 0, 0);
        }
        else
        {
            isTargetSelected = true;
        }
    }
}


void drawTargetRect(cv::Mat& frame, const cv::Rect& rect, const cv::Scalar& color)
{
    cv::line(frame, rect.tl(), rect.tl() + cv::Point(8, 0), color); // top left -
    cv::line(frame, rect.tl(), rect.tl() + cv::Point(0, 8), color); // top left |
    cv::line(frame, rect.tl() + cv::Point(rect.width - 8, 0), rect.tl() + cv::Point(rect.width, 0), color); // top right -
    cv::line(frame, rect.tl() + cv::Point(rect.width, 0), rect.tl() + cv::Point(rect.width, 8), color); // top right |
    cv::line(frame, rect.br() + cv::Point(-8, 0), rect.br(), color); // bottom right -
    cv::line(frame, rect.br() + cv::Point(0, -8), rect.br(), color); // bottom right |
    cv::line(frame, rect.tl() + cv::Point(0, rect.height), rect.tl() + cv::Point(8, rect.height), color); // bottom left -
    cv::line(frame, rect.tl() + cv::Point(0, rect.height - 8), rect.tl() + cv::Point(0, rect.height), color); // bottom left |
}


int main(int argc, char* argv[])
{
    cv::VideoCapture capture;
    if (capture.open("/home/baldman/YandexDisk/samples/videos/helicopters/7.mp4") == false)
    {
        capture.open(0);
    }
    cv::namedWindow("Output");
    cv::setMouseCallback("Output", mouseHandler);
    char key = 0;
    cv::Mat frame;
    while (key != 'q')
    {
        if (key == 32)
        {
            cv::Mat tmp;
            while (isTargetSelected == false)
            {
                tmp = frame.clone();
                char k = cv::waitKey(1);
                cv::rectangle(tmp, roi, cv::Scalar(0, 255, 0));
                cv::imshow("Output", tmp);
                if (k == 32)
                {
                    break;
                }
            }
        }
        else
        {
            if (capture.read(frame) == true)
            {
                if (isTargetSelected == true)
                {
#ifdef USE_DEBUG_INFO
                    std::cout << "=======================================" << std::endl;
                    auto begin = std::chrono::high_resolution_clock::now();
#endif // USE_LOGGING

                    roi = tracker.getTargetRect(frame, roi);

#ifdef USE_DEBUG_INFO
                    auto end = std::chrono::high_resolution_clock::now();
                    std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "\n" << std::endl;
#endif // USE_LOGGING
                }
//                cv::rectangle(frame, roi, cv::Scalar(0, 255, 0));
                drawTargetRect(frame, roi, cv::Scalar(0, 0, 255));
                cv::imshow("Output", frame);
            }
            else
            {
                break;
            }
        }
        key = cv::waitKey(20);
    }
    capture.release();
    return 0;
}
