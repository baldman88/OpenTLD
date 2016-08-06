#include "Leaf.hpp"


Leaf::Leaf()
    : positive(0), negative(0), posterior(0.0) {}


void Leaf::increment()
{
    std::lock_guard<std::mutex> lock(mutex);
    positive++;
    if (positive == 1000000000)
    {
        positive = static_cast<uint32_t>(round(positive / 1000000.0));
        negative = static_cast<uint32_t>(round(negative / 1000000.0));
    }
    posterior = static_cast<double>(positive) / (positive + negative);
}


void Leaf::decrement()
{
    std::lock_guard<std::mutex> lock(mutex);
    negative++;
    if (negative == 1000000000)
    {
        positive = static_cast<uint32_t>(round(positive / 1000000.0));
        negative = static_cast<uint32_t>(round(negative / 1000000.0));
    }
    posterior = static_cast<double>(positive) / (positive + negative);
}


double Leaf::load() const
{
    return posterior;
}


void Leaf::reset()
{
    positive = 0;
    negative = 0;
    posterior = 0.0;
}
