#include "Fern.hpp"

Fern::Fern(const int featuresCount, const double minScale)
{
    for (int feature = 0; feature < featuresCount; feature++) {
        features.push_back(new Feature(minScale));
    }
    int leafsCount = pow(4, featuresCount);
    positives = std::vector<int>(leafsCount, 0);
    negatives = std::vector<int>(leafsCount, 0);
    posteriors = std::vector<float>(leafsCount, 0.0);
}

Fern::~Fern()
{
    for (uint feature = 0; feature < features.size(); feature++) {
        delete features[feature];
    }
}

void Fern::train(const cv::Mat& frame, const cv::Rect& patch,
                 const int patchClass)
{
    int leaf = getLeafIndex(frame, patch);
    std::lock_guard<std::mutex> lock(mutex);
    if (patchClass == 0) {
        negatives[leaf]++;
    } else {
        positives[leaf]++;
    }
    if (positives.at(leaf) > 0) {
        posteriors[leaf] = (float)positives.at(leaf) / (positives.at(leaf) + negatives.at(leaf));
    }
}

float Fern::classify(const cv::Mat& frame, const cv::Rect& patch) const
{
    float posterior = posteriors.at(getLeafIndex(frame, patch));
    return posterior;
}

int Fern::getLeafIndex(const cv::Mat& frame, const cv::Rect& patch) const
{
    int width = frame.cols;
    int height = frame.rows;

    int gap = 2;
    int patchX = std::max(std::min(patch.x, width - gap), 0);
    int patchY = std::max(std::min(patch.y, height - gap), 0);

    int patchW = std::min(patch.width, width - patchX);
    int patchH = std::min(patch.height, height - patchY);

    int leaf = 0;

    for (uint feature = 0; feature < features.size(); feature++) {
        leaf |= ((features.at(feature)->test( frame, cv::Rect(patchX, patchY, patchW, patchH)) << (2 * feature)));
    }

    return leaf;
}

void Fern::reset()
{
    positives.assign(positives.size(), 0);
    negatives.assign(negatives.size(), 0);
    posteriors.assign(posteriors.size(), 0.0);
}
