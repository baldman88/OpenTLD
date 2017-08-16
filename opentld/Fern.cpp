#include "Fern.hpp"


Fern::Fern(const int featuresCount, const double minScale, const double maxScale)
{
    for (int feature = 0; feature < featuresCount; ++feature)
    {
        features.push_back(std::make_shared<Feature>(minScale, maxScale));
    }
    leafsCount = pow(4, featuresCount);
    leafs = std::vector<Leaf>(leafsCount);
}


void Fern::train(const cv::Mat &frame, const cv::Rect &patchRect, const bool isPositive)
{
    int leaf = getLeafIndex(frame, patchRect);
    if (isPositive == true)
    {
        leafs[leaf].increment();
    }
    else
    {
        leafs[leaf].decrement();
    }
}


double Fern::classify(const cv::Mat &frame, const cv::Rect &patchRect) const
{
    return leafs.at(getLeafIndex(frame, patchRect)).load();
}


int Fern::getLeafIndex(const cv::Mat &frame, const cv::Rect &patchRect) const
{
    int leaf = 0;
    int featureCounter = 0;
    for (auto feature: features)
    {
        leaf += (feature->test(frame, patchRect) << (2 * featureCounter));
        featureCounter++;
    }
    return leaf;
}


void Fern::reset()
{
    for (int leaf = 0; leaf < leafsCount; ++leaf)
    {
        leafs[leaf].reset();
    }
}
