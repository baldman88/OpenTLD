#include "Patch.hpp"


Patch::Patch(const cv::Rect &rect, const double confidence, const double overlap)
    : rect(rect), confidence(confidence), overlap(overlap) {}


bool Patch::operator<(const Patch &other) const
{
    return (confidence < other.confidence);
}


Patch &Patch::operator=(const Patch &other)
{
    if (this != &other) {
        rect = other.rect;
        confidence = other.confidence;
        overlap = other.overlap;
    }
    return *this;
}
