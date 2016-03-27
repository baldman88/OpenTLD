#include "Patch.hpp"


Patch::Patch(const cv::Rect& rect, const double confidence, const bool isOverlaps)
    : rect(rect), confidence(confidence), isOverlaps(isOverlaps) {}


bool Patch::operator<(const Patch& other) const
{
    return (confidence < other.confidence);
}


Patch& Patch::operator=(const Patch& other)
{
    if (this != &other) {
        rect = other.rect;
        confidence = other.confidence;
        isOverlaps = other.isOverlaps;
    }
    return *this;
}
