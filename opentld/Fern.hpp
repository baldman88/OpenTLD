#ifndef FERN_HPP
#define FERN_HPP

#include <vector>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>

#include "Feature.hpp"
#include "Leaf.hpp"

class Fern
{
public:
    explicit Fern(const int featuresCount, const double minScale, const double maxScale);
    ~Fern() = default;
    void train(const cv::Mat &frame, const cv::Rect &patchRect, const bool isPositive);
    double classify(const cv::Mat &frame, const cv::Rect &patchRect) const;
    void reset();

private:
    int leafsCount;
    std::vector<std::shared_ptr<Feature>> features;
    std::vector<Leaf> leafs;
    int getLeafIndex(const cv::Mat &frame, const cv::Rect &patchRect) const;
};

#endif /* FERN_HPP */
