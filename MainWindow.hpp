#ifndef MAINWINDOW_HPP
#define MAINWINDOW_HPP

#include <thread>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "TLDTracker.hpp"

class MainWindow
{
public:
    MainWindow();
    ~MainWindow();
    void run();
    void mouseHandler(int event, int x, int y, int flags);

private:
    std::thread thread;
    bool isStopped;
    TLDTracker tracker;
    cv::Rect roi;
    bool isSelectionActive;
    bool isTargetSelected;
    std::string windowName;
    char key;
    cv::Mat frame;
    cv::VideoCapture capture;
    void update();
};


#endif /* MAINWINDOW_HPP */
