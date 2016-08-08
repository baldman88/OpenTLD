#include "Detector.hpp"
#include <iostream>


Detector::Detector(std::shared_ptr<Classifier> &classifier)
    : classifier(classifier), patchRectWidth(0), patchRectHeight(0),
      frameWidth(0), frameHeight(0), varianceThreshold(0), minSideSize(16), currentZoneIndex(0) {}


void Detector::init(const cv::Mat &frame, const cv::Rect &patchRect)
{
    frameWidth = frame.cols;
    frameHeight = frame.rows;
    lastPatch = patchRect;
    setVarianceThreshold(frame, patchRect);
    if (zones.size() == 0)
    {
        int widthStep = frameWidth / 4;
        int heightStep = frameHeight / 4;
        int zoneWidth = widthStep * 3;
        int zoneHeight = heightStep * 3;
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                int startX = i * widthStep;
                int startY = j * heightStep;
                zones.push_back(cv::Rect(startX, startY, zoneWidth, zoneHeight));
            }
        }
    }
    cv::Point patchRectCenter = classifier->getRectCenter(patchRect);
    currentZoneIndex = getZoneIndex(patchRectCenter);
    std::cout << frameWidth << "; " << frameHeight << std::endl;
    for (size_t i = 0; i < zones.size(); ++i)
    {
        std::cout << "Zone_" << i << " = (" << zones[i].x << ", " << zones[i].y << ", "
                << zones[i].width << ", " << zones[i].height << ")" << std::endl;
    }
}


void Detector::setVarianceThreshold(const cv::Mat &frame, const cv::Rect &patchRect)
{
    cv::Mat integralFrame;
    cv::Mat squareIntegralFrame;
    cv::integral(frame, integralFrame, squareIntegralFrame);
    varianceThreshold = getPatchVariance(integralFrame, squareIntegralFrame, patchRect) / 2.0;
}


void Detector::detect(const cv::Mat &frame, const cv::Rect &patchRect, std::vector<Patch> &patches)
{
    std::cout << "***Detector***" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    if ((patchRect.width >= minSideSize) && (patchRect.height >= minSideSize)
        && (patchRect.width < frame.cols) && (patchRect.height < frame.rows)
        && (patchRect.x >= 0) && (patchRect.y >= 0)
        && ((patchRect.x + patchRect.width) < frame.cols)
        && ((patchRect.y + patchRect.height) < frame.rows))
    {
        currentPatch = patchRect;
    }
    else
    {
        currentPatch = cv::Rect(0, 0, 0, 0);
    }

    double minScale = 0.95;
    double maxScale = 1.05;
    double scaleStep = 0.05;
    if ((currentPatch.width >= (lastPatch.width * minScale))
            && (currentPatch.width <= (lastPatch.width * maxScale))
            && (currentPatch.height >= (lastPatch.height * minScale))
            && (currentPatch.height <= (lastPatch.height * maxScale)))
    {
        lastPatch = currentPatch;
    }
    else if ((currentPatch.width >= (lastPatch.width * 0.75))
            && (currentPatch.width <= (lastPatch.width * 1.25))
            && (currentPatch.height >= (lastPatch.height * 0.75))
            && (currentPatch.height <= (lastPatch.height * 1.25)))
    {
        currentPatch = lastPatch;
    }
    else
    {
        currentPatch = cv::Rect(0, 0, 0, 0);
        minScale = 0.9;
        maxScale = 1.1;
    }

    cv::Rect predictedPatch = filter.predict(currentPatch);
    if ((currentPatch.area() == 0)
            && (predictedPatch.width >= (lastPatch.width * minScale))
            && (predictedPatch.width <= (lastPatch.width * maxScale))
            && (predictedPatch.height >= (lastPatch.height * minScale))
            && (predictedPatch.height <= (lastPatch.height * maxScale))
            && ((predictedPatch.x + predictedPatch.width)  < frame.cols)
            && ((predictedPatch.y + predictedPatch.height) < frame.rows))
    {
        currentPatch = predictedPatch;
        std::cout << ">>> lastPatch = (" << lastPatch.x << ", " << lastPatch.y << ", "
                << lastPatch.width << ", " << lastPatch.height << ")" << std::endl;
        std::cout << ">>> predictedPatch = (" << predictedPatch.x << ", " << predictedPatch.y << ", "
                << predictedPatch.width << ", " << predictedPatch.height << ")" << std::endl;
    }

    std::set<int> widths;
    std::set<int> heights;

    for (double scale = minScale; scale <= maxScale; scale += scaleStep)
    {
        widths.insert(lastPatch.width * scale);
        heights.insert(lastPatch.height * scale);
    }

    std::vector<cv::Rect> testRects;
    double stepDevider = 20.0;
    cv::Point patchRectCenter = classifier->getRectCenter(lastPatch);

    uint64_t total = 0;

    for (auto widthIterator = widths.begin(); widthIterator != widths.end(); ++widthIterator)
    {
        int currentWidth = (*widthIterator);
        int xStep = static_cast<int>(round(currentWidth / stepDevider));
        int xCurrent = patchRectCenter.x - (currentWidth / 2);
        int xMin = std::max((xCurrent - currentWidth), 0);
        int xMax = std::min((frameWidth - currentWidth), (xCurrent + currentWidth));
        if (currentPatch.area() == 0)
        {
            xMin = zones[currentZoneIndex].x;
            xMax = xMin + zones[currentZoneIndex].width - currentWidth;
        }
        for (int x = xMin; x < xMax; x += xStep)
        {
            for (auto heightIterator = heights.begin(); heightIterator != heights.end(); ++heightIterator)
            {
                int currentHeight = (*heightIterator);
                int yCurrent = patchRectCenter.y - (currentHeight / 2);
                int yMin = std::max((yCurrent - currentHeight), 0);
                int yMax = std::min((frameHeight - currentHeight), (yCurrent + currentHeight));
                int yStep = static_cast<int>(round(currentHeight / stepDevider));
                if (currentPatch.area() == 0)
                {
                    yMin = zones[currentZoneIndex].y;
                    yMax = yMin + zones[currentZoneIndex].height - currentHeight;
                }
//                std::cout << "currentWidth = " << currentWidth << "; currentHeight = " << currentHeight << std::endl;
//                std::cout << "xMin = " << xMin << "; xMax = " << xMax << "; yMin = " << yMin << "; yMax = " << yMax << std::endl;
                for (int y = yMin; y < yMax; y += yStep)
                {
                    testRects.push_back(cv::Rect(x, y, currentWidth, currentHeight));
                    total++;
                }
            }
        }
    }
    if (currentPatch.area() > 0)
    {
        currentZoneIndex = getZoneIndex(patchRectCenter);
    }
    else
    {
        currentZoneIndex++;
        if (currentZoneIndex == zones.size())
        {
            currentZoneIndex = 0;
        }
    }
    std::cout << "currentZoneIndex = " << currentZoneIndex << std::endl;
    std::cout << "testRects.size() = " << testRects.size() << std::endl;
    std::cout << "total = " << total << std::endl;
    cv::Mat integralFrame;
    cv::Mat squareIntegralFrame;
    cv::integral(frame, integralFrame, squareIntegralFrame);
    auto end = concurrent::blockingFilter(testRects.begin(), testRects.end(),
                                          std::bind(&Detector::checkPatchVariace, this,
                                                    integralFrame, squareIntegralFrame, std::placeholders::_1));
    testRects.erase(end, testRects.end());
    std::cout << "testRects.size() = " << testRects.size() << std::endl;
    patches.resize(testRects.size());
    concurrent::blockingMapped(testRects.begin(), testRects.end(), patches.begin(),
                               std::bind(&Detector::getPatch, this, std::placeholders::_1, integralFrame, patchRect));
    if (patches.size() > 0)
    {
        auto end = concurrent::blockingFilter(patches.begin(), patches.end(),
                                              std::bind(&Detector::checkPatchConformity, this, std::placeholders::_1));
        patches.erase(end, patches.end());
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "patches.size() = " << patches.size() << std::endl;
    std::cout << "Detector elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
    std::cout << "***Detector***\n" << std::endl;
}


//void Detector::detect(const cv::Mat &frame, const cv::Rect &patchRect, std::vector<Patch> patches)
//{
//    std::cout << "***Detector***" << std::endl;
//    auto start = std::chrono::high_resolution_clock::now();
//
//    if ((patchRect.width >= minSideSize) && (patchRect.height >= minSideSize)
//        && (patchRect.width < frame.cols) && (patchRect.height < frame.rows)
//        && (patchRect.x >= 0) && (patchRect.y >= 0)
//        && ((patchRect.x + patchRect.width) < frame.cols)
//        && ((patchRect.y + patchRect.height) < frame.rows))
//    {
//        currentPatch = patchRect;
//    }
//    else
//    {
//        currentPatch = cv::Rect(0, 0, 0, 0);
//    }
//
//    if (currentPatch.area() > 0)
//    {
//        lastPatch = currentPatch;
//    }
//
//    double minScale;
//    double maxScale;
//    double scaleStep = 0.05;
//    if (currentPatch.area() > 0)
//    {
//        minScale = 0.95;
//        maxScale = 1.05;
//    }
//    else
//    {
//        minScale = 0.9;
//        maxScale = 1.1;
//    }
//
//    std::set<int> widths;
//    std::set<int> heights;
//
//    for (double scale = minScale; scale <= maxScale; scale += scaleStep)
//    {
//        widths.insert(lastPatch.width * scale);
//        heights.insert(lastPatch.height * scale);
//    }
//
//    std::vector<cv::Rect> testRects;
//    double stepDevider = 20.0;
//    cv::Point patchRectCenter = classifier->getRectCenter(lastPatch);
//    std::random_device randomDevice;
//    std::mt19937 randomEngine(randomDevice());
//
//    uint64_t total = 0;
//    if (currentPatch.area() > 0)
//    {
//        currentZoneIndex = getZoneIndex(patchRectCenter);
//    }
//
//    std::uniform_int_distribution<int> distributionX(0, std::max((frameWidth - (*(widths.begin()) * 3)), 0));
//    std::uniform_int_distribution<int> distributionY(0, std::max((frameHeight - (*(heights.begin()) * 3)), 0));
//    int xOffset = distributionX(randomEngine);
//    int yOffset = distributionY(randomEngine);
//
//    for (auto widthIterator = widths.begin(); widthIterator != widths.end(); ++widthIterator)
//    {
//        int currentWidth = (*widthIterator);
//        int xStep = static_cast<int>(round(currentWidth / stepDevider));
//        int xCurrent = patchRectCenter.x - (currentWidth / 2);
//        int xMin = std::max((xCurrent - currentWidth), 0);
//        int xMax = std::min((frameWidth - currentWidth), (xCurrent + currentWidth));
//        if (currentPatch.area() == 0)
//        {
//            std::cout << "Here!" << std::endl;
//            xMin += xOffset;
//            xMax = std::min((xMin + (currentWidth * 2)), (frameWidth - currentWidth));
//        }
//        for (int x = xMin; x < xMax; x += xStep)
//        {
//            for (auto heightIterator = heights.begin(); heightIterator != heights.end(); ++heightIterator)
//            {
//                int currentHeight = (*heightIterator);
//                int yCurrent = patchRectCenter.y - (currentHeight / 2);
//                int yMin = std::max((yCurrent - currentHeight), 0);
//                int yMax = std::min((frameHeight - currentHeight), (yCurrent + currentHeight));
//                int yStep = static_cast<int>(round(currentHeight / stepDevider));
//                if (currentPatch.area() == 0)
//                {
//                    yMin += yOffset;
//                    yMax = std::min((yMin + (currentHeight * 2)), (frameHeight - currentHeight));
//                }
//                for (int y = yMin; y < yMax; y += yStep)
//                {
//                    testRects.push_back(cv::Rect(x, y, currentWidth, currentHeight));
//                    total++;
//                }
//            }
//        }
//    }
//    std::cout << "xOffset = " << xOffset << "; yOffset = " << yOffset << std::endl;
//    std::cout << "testRects.size() = " << testRects.size() << std::endl;
//    std::cout << "total = " << total << std::endl;
//    cv::Mat integralFrame;
//    cv::Mat squareIntegralFrame;
//    cv::integral(frame, integralFrame, squareIntegralFrame);
//    auto end = concurrent::blockingFilter(testRects.begin(), testRects.end(),
//                                          std::bind(&Detector::checkPatchVariace, this,
//                                                    integralFrame, squareIntegralFrame, std::placeholders::_1));
//    testRects.erase(end, testRects.end());
//    patches.resize(testRects.size());
//    concurrent::blockingMapped(testRects.begin(), testRects.end(), patches.begin(),
//                               std::bind(&Detector::getPatch, this, std::placeholders::_1, integralFrame, patchRect));
//    if (patches.size() > 0)
//    {
//        auto end = concurrent::blockingFilter(patches.begin(), patches.end(),
//                                              std::bind(&Detector::checkPatchConformity, this, std::placeholders::_1));
//        patches.erase(end, patches.end());
//    }
//    auto stop = std::chrono::high_resolution_clock::now();
//    std::cout << "patches.size() = " << patches.size() << std::endl;
//    std::cout << "Detector elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
//    std::cout << "***Detector***\n" << std::endl;
//}


//void Detector::detect(const cv::Mat &frame, const cv::Rect &patchRect, std::vector<Patch> patches)
//{
//    std::cout << "***Detector***" << std::endl;
//    auto start = std::chrono::high_resolution_clock::now();
//
//    if ((patchRect.width >= minSideSize) && (patchRect.height >= minSideSize)
//        && (patchRect.width < frame.cols) && (patchRect.height < frame.rows)
//        && (patchRect.x >= 0) && (patchRect.y >= 0)
//        && ((patchRect.x + patchRect.width) < frame.cols)
//        && ((patchRect.y + patchRect.height) < frame.rows))
//    {
//        currentPatch = patchRect;
//    }
//    else
//    {
//        currentPatch = cv::Rect(0, 0, 0, 0);
//    }
//
//    if (currentPatch.area() > 0)
//    {
//        lastPatch = currentPatch;
//    }
//
//    double minScale;
//    double maxScale;
//    double scaleStep = 0.05;
//    if (currentPatch.area() > 0)
//    {
//        minScale = 0.95;
//        maxScale = 1.05;
//    }
//    else
//    {
//        minScale = 0.9;
//        maxScale = 1.1;
//    }
//
//    std::set<int> widths;
//    std::set<int> heights;
//
//    for (double scale = minScale; scale <= maxScale; scale += scaleStep)
//    {
//        widths.insert(lastPatch.width * scale);
//        heights.insert(lastPatch.height * scale);
//    }
//
//    uint64_t total = 0;
//
//    std::vector<cv::Rect> testRects;
//    double stepDevider = 20.0;
//    cv::Point patchRectCenter = classifier->getRectCenter(lastPatch);
//    std::random_device randomDevice;
//    std::mt19937 randomEngine(randomDevice());
//
//    for (auto widthIterator = widths.begin(); widthIterator != widths.end(); ++widthIterator) {
//        int currentWidth = (*widthIterator);
//        int xMin;
//        int xMax;
//        int xStep = static_cast<int>(round(currentWidth / stepDevider));
//        if (currentPatch.area() > 0) {
//            int xCurrent = patchRectCenter.x - (currentWidth / 2);
//            xMin = std::max((xCurrent - currentWidth), 0);
//            xMax = std::min((frameWidth - currentWidth), (xCurrent + currentWidth));
//        } else {
//            std::uniform_int_distribution<int> distributionX(0, std::max((frameWidth - (currentWidth * 3)), 0));
//            xMin = distributionX(randomEngine);
//            xMax = std::min((xMin + (currentWidth * 2)), (frameWidth - currentWidth));
//        }
//        for (int x = xMin; x < xMax; x += xStep) {
//            for (auto heightIterator = heights.begin(); heightIterator != heights.end(); ++heightIterator) {
//                int currentHeight = (*heightIterator);
//                int yMin;
//                int yMax;
//                int yStep = static_cast<int>(round(currentHeight / stepDevider));
//                if (currentPatch.area() > 0) {
//                    int yCurrent = patchRectCenter.y - (currentHeight / 2);
//                    yMin = std::max((yCurrent - currentHeight), 0);
//                    yMax = std::min((frameHeight - currentHeight), (yCurrent + currentHeight));
//                } else {
//                    std::uniform_int_distribution<int> distributionY(0, std::max((frameHeight - (currentHeight * 3)), 0));
//                    yMin = distributionY(randomEngine);
//                    yMax = std::min((yMin + (currentHeight * 2)), (frameHeight - currentHeight));
//                }
//                for (int y = yMin; y < yMax; y += yStep) {
//                    testRects.push_back(cv::Rect(x, y, currentWidth, currentHeight));
//                    total++;
//                }
//            }
//        }
//    }
//    std::cout << "testRects.size() = " << testRects.size() << std::endl;
//    std::cout << "total = " << total << std::endl;
//    cv::Mat integralFrame;
//    cv::Mat squareIntegralFrame;
//    cv::integral(frame, integralFrame, squareIntegralFrame);
//    auto end = concurrent::blockingFilter(testRects.begin(), testRects.end(),
//                                          std::bind(&Detector::checkPatchVariace, this,
//                                                    integralFrame, squareIntegralFrame, std::placeholders::_1));
//    testRects.erase(end, testRects.end());
//    patches.resize(testRects.size());
//    concurrent::blockingMapped(testRects.begin(), testRects.end(), patches.begin(),
//                               std::bind(&Detector::getPatch, this, std::placeholders::_1, integralFrame, patchRect));
//    if (patches.size() > 0)
//    {
//        auto end = concurrent::blockingFilter(patches.begin(), patches.end(),
//                                              std::bind(&Detector::checkPatchConformity, this, std::placeholders::_1));
//        patches.erase(end, patches.end());
//    }
//    auto stop = std::chrono::high_resolution_clock::now();
//    std::cout << "patches.size() = " << patches.size() << std::endl;
//    std::cout << "Detector elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << std::endl;
//    std::cout << "***Detector***\n" << std::endl;
//}


double Detector::getPatchVariance(const cv::Mat &integralFrame, const cv::Mat &squareIntegralFrame, const cv::Rect &patchRect) const
{
    double variance = 0;
    double area = patchRect.area();
    if (area > 0)
    {
        double mean = (integralFrame.at<int>(cv::Point(patchRect.x, patchRect.y))
                      + integralFrame.at<int>(cv::Point(patchRect.x + patchRect.width, patchRect.y + patchRect.height))
                      - integralFrame.at<int>(cv::Point(patchRect.x + patchRect.width, patchRect.y))
                      - integralFrame.at<int>(cv::Point(patchRect.x, patchRect.y + patchRect.height))) / area;
        double deviance = (squareIntegralFrame.at<double>(cv::Point(patchRect.x, patchRect.y))
                          + squareIntegralFrame.at<double>(cv::Point(patchRect.x + patchRect.width, patchRect.y + patchRect.height))
                          - squareIntegralFrame.at<double>(cv::Point(patchRect.x + patchRect.width, patchRect.y))
                          - squareIntegralFrame.at<double>(cv::Point(patchRect.x, patchRect.y + patchRect.height))) / area;
        variance = deviance - (mean * mean);
    }
    return variance;
}


Patch Detector::getPatch(const cv::Rect &testRect, const cv::Mat &frame, const cv::Rect &patchRect) const
{
    double confidence = classifier->classify(frame, testRect);
    bool overlap = false;
    if ((patchRect.area() > 0) && (classifier->getRectsOverlap(patchRect, testRect) > classifier->minOverlap))
    {
        overlap = true;
    }
    return Patch(testRect, confidence, overlap);
}


bool Detector::checkPatchConformity(const Patch &patch) const
{
    bool result = false;
    if ((patch.confidence > 0.6f) || (patch.isOverlaps == true))
    {
        result = true;
    }
    return result;
}


bool Detector::checkPatchVariace(const cv::Mat &integralFrame, const cv::Mat &squareIntegralFrame, const cv::Rect &patchRect) const
{
    bool result = false;
    if (getPatchVariance(integralFrame, squareIntegralFrame, patchRect) > varianceThreshold)
    {
        result = true;
    }
    return result;
}


int Detector::getZoneIndex(const cv::Point &point) const
{
    int zoneIndex = 0;
    int distance = 10000;
    for (size_t i = 0; i < zones.size(); ++i)
    {
        cv::Point zoneCenter = classifier->getRectCenter(zones[i]);
        if (distance > getDistance(zoneCenter, point))
        {
            distance = getDistance(zoneCenter, point);
            zoneIndex = i;
        }
    }
    return zoneIndex;
}


int Detector::getDistance(const cv::Point &first, const cv::Point &second) const
{
    return round(sqrt(pow((first.x - second.x), 2) + pow((first.y - second.y), 2)));
}
