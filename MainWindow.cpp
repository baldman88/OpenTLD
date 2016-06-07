#include "MainWindow.hpp"
#include <iostream>


void mouseHandlerWrapper(int event, int x, int y, int flags, void* param)
{
    std::cout << "mouseHandlerWrapper()" << std::endl;
    MainWindow* mw = static_cast<MainWindow*>(param);
    mw->mouseHandler(event, x, y, flags);
}


MainWindow::MainWindow()
    : thread(), isStopped(false), tracker(), roi(0, 0, 0, 0),
      isSelectionActive(false), isTargetSelected(false),
      windowName("Output"), key(0), frame(), capture()
{
    if (capture.open(1) == false)
    {
        capture.open(0);
    }
    cv::namedWindow(windowName);
    cv::setMouseCallback(windowName, mouseHandlerWrapper, this);
}


MainWindow::~MainWindow()
{
    isStopped = true;
//    key = 'q';
    cv::destroyWindow(windowName);
    if (thread.joinable() == true)
    {
        thread.join();
    }
}


void MainWindow::run()
{
    thread = std::thread(&MainWindow::update, this);
}


void MainWindow::update()
{
    std::cout << "MainWindow::update()" << std::endl;
    while (key != 'q')
    {
        capture.read(frame);
        if (isTargetSelected == true)
        {
            roi = tracker.getTargetRect(frame, roi);
        }
        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0));
        cv::imshow(windowName, frame);
        key = cv::waitKey(1);
    }
}


void MainWindow::mouseHandler(int event, int x, int y, int flags)
{
    std::cout << "MainWindow::mouseHandler()" << std::endl;
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
