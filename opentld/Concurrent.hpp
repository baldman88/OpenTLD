#ifndef CONCURRENT_HPP
#define CONCURRENT_HPP

#include <cmath>
#include <thread>
#include <vector>
#include <functional>


namespace concurrent
{
    template <typename InputIterator, typename MapFunctor>
    void mapHelper(InputIterator first, InputIterator last, MapFunctor mapFunctor)
    {
        while (first != last) {
            mapFunctor(*first);
            ++first;
        }
    }

    template<typename InputIterator, typename OutputIterator, typename MapFunctor>
    OutputIterator mappedHelper(InputIterator first, InputIterator last, OutputIterator result, MapFunctor mapFunctor)
    {
        while (first != last)
        {
            *result = mapFunctor(*first);
            ++result;
            ++first;
        }
        return result;
    }

    template<typename InputIterator, typename MapFunctor>
    void blockingMap(InputIterator first, InputIterator last, MapFunctor mapFunctor)
    {
        auto availableThreads = std::thread::hardware_concurrency();
        if (availableThreads < 2) {
            mapHelper(first, last, mapFunctor);
        } else {
            const auto totalSize = std::distance(first, last);
            const auto blockSize = static_cast<size_t>(std::ceil(1.0 * totalSize / availableThreads));
            std::vector<std::thread> threads;
            auto blockStart = first;
            auto blockEnd = first;
            for (size_t i = 0; i < (availableThreads - 1); ++i)
            {
                std::advance(blockEnd, blockSize);
                threads.emplace_back(std::thread(mapHelper<InputIterator, MapFunctor>,
                                                 blockStart, blockEnd, mapFunctor));
                blockStart = blockEnd;
            }
            mapHelper(blockStart, last, mapFunctor);
            for (size_t i = 0; i < threads.size(); ++i)
            {
                if (threads[i].joinable())
                {
                    threads[i].join();
                }
            }
        }
    }


    template<typename InputIterator, typename OutputIterator, typename MapFunctor>
    OutputIterator blockingMapped(InputIterator first, InputIterator last, OutputIterator result, MapFunctor mapFunctor)
    {
        auto availableThreads = std::thread::hardware_concurrency();
        if (availableThreads < 2)
        {
            mappedHelper(first, last, result, mapFunctor);
        }
        else
        {
            const auto totalSize = std::distance(first, last);
            const auto blockSize = static_cast<size_t>(std::ceil(1.0 * totalSize / availableThreads));
            std::vector<std::thread> threads;
            auto blockStart = first;
            auto blockEnd = first;
            for (size_t i = 0; i < (availableThreads - 1); ++i)
            {
                std::advance(blockEnd, blockSize);
                threads.emplace_back(std::thread(mappedHelper<InputIterator, OutputIterator, MapFunctor>,
                                                 blockStart, blockEnd, result, mapFunctor));
                blockStart = blockEnd;
                std::advance(result, blockSize);
            }
            result = mappedHelper(blockStart, last, result, mapFunctor);
            for (size_t i = 0; i < threads.size(); ++i)
            {
                if (threads[i].joinable())
                {
                    threads[i].join();
                }
            }
        }
        return result;
    }


    template<typename InputIterator, typename FilterFunctor>
    InputIterator blockingFilter(InputIterator first, InputIterator last, FilterFunctor filterFunctor)
    {
        auto result = first;
//    auto availableThreads = std::thread::hardware_concurrency() / 2;
//    if (availableThreads < 2) {
        while (first != last) {
            if (filterFunctor(*first) == true) {
                *result = *first;
                ++result;
            }
            ++first;
        }
//    } else {
//        std::deque<std::pair<InputIterator, std::future<bool>>> futures;
//        while (first != last) {
//            for (auto thread = futures.size(); ((thread < availableThreads) && (first != last)); ++thread) {
//                futures.push_back(std::pair<InputIterator, std::future<bool>>(first, std::move(std::async(std::launch::async, filterFunctor, *first))));
//                ++first;
//            }
//            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
//                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//                    if ((*futureIterator).second.get() == true) {
//                        *result = *((*futureIterator).first);
//                        ++result;
//                    }
//                    futureIterator = futures.erase(futureIterator);
//                } else {
//                    ++futureIterator;
//                }
//            }
//        }
//        while (futures.size() > 0) {
//            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
//                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//                    if ((*futureIterator).second.get() == true) {
//                        *result = *((*futureIterator).first);
//                        ++result;
//                    }
//                    futureIterator = futures.erase(futureIterator);
//                } else {
//                    ++futureIterator;
//                }
//            }
//        }
//    }
        return result;
    }


    template<typename InputIterator, typename OutputIterator, typename FilterFunctor>
    OutputIterator blockingFiltered(InputIterator first, InputIterator last, OutputIterator result, FilterFunctor filterFunctor)
    {
//    auto availableThreads = std::thread::hardware_concurrency() / 2;
//    if (availableThreads < 2) {
        while (first != last) {
            if (filterFunctor(*first) == true) {
                *result = *first;
                ++result;
            }
            ++first;
        }
//    } else {
//        std::deque<std::pair<InputIterator, std::future<bool>>> futures;
//        while (first != last) {
//            for (auto thread = futures.size(); ((thread < availableThreads) && (first != last)); ++thread) {
//                futures.push_back(std::pair<InputIterator, std::future<bool>>(first, std::move(std::async(std::launch::async, filterFunctor, *first))));
//                ++first;
//            }
//            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
//                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//                    if ((*futureIterator).second.get() == true) {
//                        *result = *((*futureIterator).first);
//                        ++result;
//                    }
//                    futureIterator = futures.erase(futureIterator);
//                } else {
//                    ++futureIterator;
//                }
//            }
//        }
//        while (futures.size() > 0) {
//            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
//                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//                    if ((*futureIterator).second.get() == true) {
//                        *result = *((*futureIterator).first);
//                        ++result;
//                    }
//                    futureIterator = futures.erase(futureIterator);
//                } else {
//                    ++futureIterator;
//                }
//            }
//        }
//    }
        return result;
    }
}


#endif /* CONCURRENT_HPP */
