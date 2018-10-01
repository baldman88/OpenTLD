#ifndef CONCURRENT_HPP
#define CONCURRENT_HPP

#include <cmath>
#include <vector>
#include <future>
#include <algorithm>
#include <functional>


namespace concurrent
{
    template <typename InputIterator, typename MapFunctor>
    void for_each(InputIterator first, InputIterator last, MapFunctor mapFunctor)
    {
        while (first != last) {
            mapFunctor(*first);
            ++first;
        }
    }

    template<typename InputIterator, typename OutputIterator, typename MapFunctor>
    OutputIterator for_each(InputIterator first, InputIterator last, OutputIterator result, MapFunctor mapFunctor)
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
            mapFunctor(first, last);
        } else {
            auto const totalSize = std::distance(first, last);
            auto const blockSize = totalSize / availableThreads;
            std::vector<std::thread> threads(availableThreads - 1);
            InputIterator blockStart = first;
            for (unsigned int i = 0; i < (availableThreads - 1); ++i) {
                InputIterator blockEnd = blockStart;
                std::advance(blockEnd, blockSize);
                threads[i] = std::thread(mapFunctor, blockStart, blockEnd);
                blockStart = blockEnd;
            }
            mapFunctor(blockStart, last);
            std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
        }
    }


    template<typename InputIterator, typename OutputIterator, typename MapFunctor>
    OutputIterator blockingMapped(InputIterator first, InputIterator last, OutputIterator result, MapFunctor mapFunctor)
    {
        auto availableThreads = std::thread::hardware_concurrency();
        if (availableThreads < 2)
        {
            for_each(first, last, result, mapFunctor);
        }
        else
        {
            auto const totalSize = std::distance(first, last);
            auto const blockSize = static_cast<size_t>(std::ceil(1.0 * totalSize / availableThreads));
            std::vector<std::thread> threads;
            using InputType = typename std::iterator_traits<InputIterator>::value_type;
            using OutputType = typename std::result_of<MapFunctor(InputType)>::type;
            std::vector<std::vector<OutputType>> results;
            for (size_t i = 0; i < (availableThreads - 1); ++i)
            {
                results.emplace_back(std::vector<OutputType>(blockSize));
            }
            InputIterator current = first;
            for (unsigned int i = 0; i < (availableThreads - 1); ++i)
            {
                InputIterator blockStart = current;
                InputIterator blockEnd = blockStart;
                std::advance(blockEnd, blockSize);
                current = blockEnd;
                threads.emplace_back(std::thread(for_each<InputIterator, OutputIterator, MapFunctor>,
                                                 blockStart, blockEnd, results.at(i).begin(), mapFunctor));
            }
            result = for_each(current, last, result, mapFunctor);
            for (unsigned int i = 0; i < threads.size(); ++i)
            {
                if (threads[i].joinable())
                {
                    threads[i].join();
                }
                result = std::copy(results.at(i).begin(), results.at(i).end(), result);
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
