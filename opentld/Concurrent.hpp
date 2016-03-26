#ifndef CONCURRENT_HPP
#define CONCURRENT_HPP

#include <map>
#include <deque>
#include <future>
#include <chrono>
#include <iterator>
#include <type_traits>


namespace concurrent
{
template<class InputIterator, class MapFunctor>
InputIterator blockingMap(InputIterator first, InputIterator last, MapFunctor mapFunctor)
{
    auto result = first;
    auto availableThreads = std::thread::hardware_concurrency() / 2;
    if (availableThreads < 2) {
        while (result != last) {
            mapFunctor(std::ref(*result));
            ++result;
        }
    } else {
        std::deque<std::future<void>> futures;
        while (result != last) {
            for (auto thread = futures.size(); ((thread < availableThreads) && (result != last)); ++thread) {
                futures.push_back(std::move(std::async(std::launch::async, mapFunctor, std::ref(*result))));
                ++result;
            }
            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
                if (futureIterator->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    futureIterator->get();
                    futureIterator = futures.erase(futureIterator);
                } else {
                    ++futureIterator;
                }
            }
        }
        while (futures.size() > 0) {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
                if (futureIterator->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    futureIterator->get();
                    futureIterator = futures.erase(futureIterator);
                } else {
                    ++futureIterator;
                }
            }
        }
    }
    return result;
}


template<class InputIterator, class OutputIterator, class MapFunctor>
OutputIterator blockingMapped(InputIterator first, InputIterator last, OutputIterator result, MapFunctor mapFunctor)
{
    auto availableThreads = std::thread::hardware_concurrency() / 2;
    if (availableThreads < 2) {
        while (first != last) {
            *result = mapFunctor(*first);
            ++result;
            ++first;
        }
    } else {
        using InputType = typename std::iterator_traits<InputIterator>::value_type;
        using OutputType = typename std::result_of<MapFunctor(InputType)>::type;
        std::deque<std::future<OutputType>> futures;
        while (first != last) {
            for (auto thread = futures.size(); ((thread < availableThreads) && (first != last)); ++thread) {
                futures.push_back(std::move(std::async(std::launch::async, mapFunctor, *first)));
                ++first;
            }
            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
                if ((*futureIterator).wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    *result = (*futureIterator).get();
                    ++result;
                    futureIterator = futures.erase(futureIterator);
                } else {
                    ++futureIterator;
                }
            }
        }
        while (futures.size() > 0) {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
                if ((*futureIterator).wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    *result = (*futureIterator).get();
                    ++result;
                    futureIterator = futures.erase(futureIterator);
                } else {
                    ++futureIterator;
                }
            }
        }
    }
    return result;
}


template<class InputIterator, class FilterFunctor>
InputIterator blockingFilter(InputIterator first, InputIterator last, FilterFunctor filterFunctor)
{
    auto result = first;
    auto availableThreads = std::thread::hardware_concurrency() / 2;
    if (availableThreads < 2) {
        while (first != last) {
            if (filterFunctor(*first) == true) {
                *result = *first;
                ++result;
            }
            ++first;
        }
    } else {
        std::deque<std::pair<InputIterator, std::future<bool>>> futures;
        while (first != last) {
            for (auto thread = futures.size(); ((thread < availableThreads) && (first != last)); ++thread) {
                futures.push_back(std::pair<InputIterator, std::future<bool>>(first, std::move(std::async(std::launch::async, filterFunctor, *first))));
                ++first;
            }
            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    if ((*futureIterator).second.get() == true) {
                        *result = *((*futureIterator).first);
                        ++result;
                    }
                    futureIterator = futures.erase(futureIterator);
                } else {
                    ++futureIterator;
                }
            }
        }
        while (futures.size() > 0) {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    if ((*futureIterator).second.get() == true) {
                        *result = *((*futureIterator).first);
                        ++result;
                    }
                    futureIterator = futures.erase(futureIterator);
                } else {
                    ++futureIterator;
                }
            }
        }
    }
    return result;
}


template<class InputIterator, class OutputIterator, class FilterFunctor>
OutputIterator blockingFiltered(InputIterator first, InputIterator last, OutputIterator result, FilterFunctor filterFunctor)
{
    auto availableThreads = std::thread::hardware_concurrency() / 2;
    if (availableThreads < 2) {
        while (first != last) {
            if (filterFunctor(*first) == true) {
                *result = *first;
                ++result;
            }
            ++first;
        }
    } else {
        std::deque<std::pair<InputIterator, std::future<bool>>> futures;
        while (first != last) {
            for (auto thread = futures.size(); ((thread < availableThreads) && (first != last)); ++thread) {
                futures.push_back(std::pair<InputIterator, std::future<bool>>(first, std::move(std::async(std::launch::async, filterFunctor, *first))));
                ++first;
            }
            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    if ((*futureIterator).second.get() == true) {
                        *result = *((*futureIterator).first);
                        ++result;
                    }
                    futureIterator = futures.erase(futureIterator);
                } else {
                    ++futureIterator;
                }
            }
        }
        while (futures.size() > 0) {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end();) {
                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    if ((*futureIterator).second.get() == true) {
                        *result = *((*futureIterator).first);
                        ++result;
                    }
                    futureIterator = futures.erase(futureIterator);
                } else {
                    ++futureIterator;
                }
            }
        }
    }
    return result;
}
}


#endif /* CONCURRENT_HPP */
