#include <iostream>

#include <opencv2/highgui.hpp>

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
        isTargetSelected = true;
    }
}


int main(int argc, char* argv[])
{
    cv::VideoCapture capture;
    if (capture.open(1) == false)
    {
        capture.open(0);
    }
    cv::namedWindow("Output");
    cv::setMouseCallback("Output", mouseHandler);
    char key = 0;
    cv::Mat frame;
    while (key != 'q')
    {
        capture.read(frame);
        if (isTargetSelected == true)
        {
            roi = tracker.getTargetRect(frame, roi);
        }
        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0));
        cv::imshow("Output", frame);
        key = cv::waitKey(1);
    }
    return 0;
}
