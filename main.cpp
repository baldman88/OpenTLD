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
        if ((roi.width < 20) || (roi.height < 20))
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
    if (capture.open(1) == false)
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
        capture.read(frame);
        if (isTargetSelected == true)
        {
            auto begin = std::chrono::high_resolution_clock::now();
            roi = tracker.getTargetRect(frame, roi);
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        }
        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0));
        cv::imshow("Output", frame);
        key = cv::waitKey(20);
    }
    capture.release();
    return 0;
}
