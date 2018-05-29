#ifndef CONCURRENT_HPP
#define CONCURRENT_HPP

#include <map>
#include <deque>
#include <vector>
#include <future>
#include <chrono>
#include <iterator>
#include <algorithm>
#include <type_traits>


namespace concurrent
{
template<class InputIterator, class MapFunctor>
void blockingMap(InputIterator first, InputIterator last, MapFunctor mapFunctor)
{
    auto availableThreads = std::thread::hardware_concurrency() / 2;
    if (availableThreads == 0) {
        availableThreads = 2;
    }
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


template<class InputIterator, class OutputIterator, class MapFunctor>
OutputIterator blockingMapped(InputIterator first, InputIterator last, OutputIterator result, MapFunctor mapFunctor)
{
//    auto availableThreads = std::thread::hardware_concurrency() / 2;
//    if (availableThreads < 2) {
        while (first != last) {
            *result = mapFunctor(*first);
            ++result;
            ++first;
        }
//    } else {
//        using InputType = typename std::iterator_traits<InputIterator>::value_type;
//        using OutputType = typename std::result_of<MapFunctor(InputType)>::type;
//        std::deque<std::future<OutputType>> futures;
//        while (first != last) {
//            for (auto thread = futures.size(); ((thread < availableThreads) && (first != last)); ++thread) {
//                futures.push_back(std::move(std::async(std::launch::async, mapFunctor, *first)));
//                ++first;
//            }
//            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
//                if ((*futureIterator).wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//                    *result = (*futureIterator).get();
//                    ++result;
//                    futureIterator = futures.erase(futureIterator);
//                } else {
//                    ++futureIterator;
//                }
//            }
//        }
//        while (futures.size() > 0) {
//            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
//                if ((*futureIterator).wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
//                    *result = (*futureIterator).get();
//                    ++result;
//                    futureIterator = futures.erase(futureIterator);
//                } else {
//                    ++futureIterator;
//                }
//            }
//        }
//    }
    return result;
}


template<class InputIterator, class FilterFunctor>
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


template<class InputIterator, class OutputIterator, class FilterFunctor>
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
