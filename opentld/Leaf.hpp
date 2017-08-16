#ifndef LEAF_HPP
#define LEAF_HPP

#include <mutex>
#include <memory>
#include <cstdint>
#include <cmath>

#include <iostream>


class Leaf
{
public:
    explicit Leaf();
    ~Leaf() = default;
    void increment();
    void decrement();
    double load() const;
    void reset();

private:
    std::mutex mutex;
    uint32_t positive;
    uint32_t negative;
    double posterior;
};


#endif /* LEAF_HPP */
