#include <iostream>
#include <chrono>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "opentld/TLDTracker.hpp"


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


int main(int argc, char* argv[])
{
    cv::VideoCapture capture;
#ifdef DEBUG
    std::cout << "This is debug!" << std::endl;
#endif
    if (capture.open("/home/baldman/YandexDisk/samples/videos/helicopters/9.mp4") == false)
    {
        capture.open(0);
    }
    std::cout << capture.get(CV_CAP_PROP_FPS) << std::endl;
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
                    auto begin = std::chrono::high_resolution_clock::now();
                    roi = tracker.getTargetRect(frame, roi);
                    auto end = std::chrono::high_resolution_clock::now();
                    std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
                }
                cv::rectangle(frame, roi, cv::Scalar(0, 255, 0));
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
