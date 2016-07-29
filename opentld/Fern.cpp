#include "Fern.hpp"


Fern::Fern(const int featuresCount, const double minScale, const double maxScale)
{
    for (int feature = 0; feature < featuresCount; ++feature)
    {
        features.push_back(std::make_shared<Feature>(minScale, maxScale));
    }
    leafsCount = pow(4, featuresCount);
    positives = std::vector<std::atomic<int>>(leafsCount);
    negatives = std::vector<std::atomic<int>>(leafsCount);
    posteriors = std::vector<std::atomic<double>>(leafsCount);
}


void Fern::train(const cv::Mat &frame, const cv::Rect &patchRect, const bool isPositive)
{
    std::cout << "In Fern::train ..." << std::endl;
    int leaf = getLeafIndex(frame, patchRect);
    std::lock_guard<std::mutex> lock(mutex);
    if (isPositive == true)
    {
        (negatives[leaf]).store(negatives[leaf].load()++);
    }
    else
    {
        (positives[leaf]).store(positives[leaf].load()++);
    }

    if (positives.at(leaf).load() > 0)
    {
        (posteriors[leaf]).store(static_cast<double>(positives.at(leaf).load()) / (positives.at(leaf).load() + negatives.at(leaf).load()));
    }
}


double Fern::classify(const cv::Mat &frame, const cv::Rect &patchRect) const
{
    return posteriors.at(getLeafIndex(frame, patchRect)).load();
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
    positives = std::vector<std::atomic<int>>(leafsCount);
    negatives = std::vector<std::atomic<int>>(leafsCount);
    posteriors = std::vector<std::atomic<double>>(leafsCount);
}
