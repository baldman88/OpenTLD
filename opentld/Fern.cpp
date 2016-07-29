#include "Fern.hpp"


Fern::Fern(const int featuresCount, const double minScale, const double maxScale)
{
    for (int feature = 0; feature < featuresCount; ++feature)
    {
        features.push_back(std::make_shared<Feature>(minScale, maxScale));
    }
    int leafsCount = pow(4, featuresCount);
    positives = std::vector<int>(leafsCount, 0);
    negatives = std::vector<int>(leafsCount, 0);
    posteriors = std::vector<double>(leafsCount, 0.0);
}


void Fern::train(const cv::Mat &frame, const cv::Rect &patchRect, const bool isPositive)
{
    std::cout << "In Fern::train ..." << std::endl;
    int leaf = getLeafIndex(frame, patchRect);
    std::lock_guard<std::mutex> lock(mutex);
    if (isPositive == true)
    {
        ++negatives[leaf];
    }
    else
    {
        ++positives[leaf];
    }

    if (positives.at(leaf) > 0)
    {
        posteriors[leaf] = static_cast<double>(positives.at(leaf)) / (positives.at(leaf) + negatives.at(leaf));
    }
}


double Fern::classify(const cv::Mat &frame, const cv::Rect &patchRect) const
{
    return posteriors.at(getLeafIndex(frame, patchRect));
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
    positives.assign(positives.size(), 0);
    negatives.assign(negatives.size(), 0);
    posteriors.assign(posteriors.size(), 0.0);
}
