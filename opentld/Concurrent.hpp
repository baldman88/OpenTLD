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
    int availableThreads = std::thread::hardware_concurrency() - 2;
    if (availableThreads < 2)
    {
        while (result != last)
        {
            mapFunctor(std::ref(*result));
            ++result;
        }
    }
    else
    {
        std::vector<std::future<void>> futures;
        for (int thread = 0; ((thread < availableThreads) && (result != last)); ++thread)
        {
            futures.push_back(std::move(std::async(std::launch::async, mapFunctor, std::ref(*result))));
            ++result;
        }
        while (result != last)
        {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end(); ++futureIterator)
            {
                if (futureIterator->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
                {
                    futureIterator->get();
                    *futureIterator = std::move(std::async(std::launch::async, mapFunctor, std::ref(*result)));
                    ++result;
                }
            }
        }
        while (futures.size() > 0)
        {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end(); ++futureIterator)
            {
                if (futureIterator->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
                {
                    futureIterator->get();
                    futureIterator = futures.erase(futureIterator);
                }
            }
        }
    }
    return result;
}


template<class InputIterator, class OutputIterator, class MapFunctor>
OutputIterator blockingMapped(InputIterator first, InputIterator last, OutputIterator result, MapFunctor mapFunctor)
{
    int availableThreads = std::thread::hardware_concurrency() - 2;
    if (availableThreads < 2)
    {
        while (first != last)
        {
            *result = mapFunctor(*first);
            ++result;
            ++first;
        }
    }
    else
    {
        using InputType = typename std::iterator_traits<InputIterator>::value_type;
        using OutputType = typename std::result_of<MapFunctor(InputType)>::type;
        std::vector<std::future<OutputType>> futures;
        for (int thread = 0; ((thread < availableThreads) && (first != last)); ++thread)
        {
            futures.push_back(std::move(std::async(std::launch::async, mapFunctor, *first)));
            ++first;
        }
        while (first != last)
        {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end(); ++futureIterator)
            {
                if ((*futureIterator).wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
                {
                    *result = (*futureIterator).get();
                    ++result;
                    *futureIterator = std::move(std::async(std::launch::async, mapFunctor, *first));
                }
            }
        }
        while (futures.size() > 0)
        {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end(); ++futureIterator)
            {
                if ((*futureIterator).wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
                {
                    *result = (*futureIterator).get();
                    ++result;
                    futureIterator = futures.erase(futureIterator);
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
    int availableThreads = std::thread::hardware_concurrency() - 2;
    if (availableThreads < 2)
    {
        while (first != last)
        {
            if (filterFunctor(*first) == true)
            {
                *result = *first;
                ++result;
            }
            ++first;
        }
    }
    else
    {
        std::vector<std::pair<InputIterator, std::future<bool>>> futures;
        for (int thread = 0; ((thread < availableThreads) && (first != last)); ++thread)
        {
            futures.push_back(std::pair<InputIterator, std::future<bool>>(first, std::move(std::async(std::launch::async, filterFunctor, *first))));
            ++first;
        }
        while (first != last)
        {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end(); ++futureIterator)
            {
                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
                {
                    if ((*futureIterator).second.get() == true)
                    {
                        *result = *((*futureIterator).first);
                        ++result;
                    }
                    *futureIterator = std::pair<InputIterator, std::future<bool>>(first, std::move(std::async(std::launch::async, filterFunctor, *first)));
                }
            }
        }
        while (futures.size() > 0)
        {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end(); ++futureIterator)
            {
                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
                {
                    if ((*futureIterator).second.get() == true)
                    {
                        *result = *((*futureIterator).first);
                        ++result;
                    }
                    futureIterator = futures.erase(futureIterator);
                }
            }
        }
    }
    return result;
}


template<class InputIterator, class OutputIterator, class FilterFunctor>
OutputIterator blockingFiltered(InputIterator first, InputIterator last, OutputIterator result, FilterFunctor filterFunctor)
{
    auto availableThreads = std::thread::hardware_concurrency() - 2;
    if (availableThreads < 2)
    {
        while (first != last)
        {
            if (filterFunctor(*first) == true)
            {
                *result = *first;
                ++result;
            }
            ++first;
        }
    }
    else
    {
        std::vector<std::pair<InputIterator, std::future<bool>>> futures;
        for (int thread = 0; ((thread < availableThreads) && (first != last)); ++thread)
        {
            futures.push_back(std::pair<InputIterator, std::future<bool>>(first, std::move(std::async(std::launch::async, filterFunctor, *first))));
            ++first;
        }
        while (first != last)
        {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end(); ++futureIterator)
            {
                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
                {
                    if ((*futureIterator).second.get() == true)
                    {
                        *result = *((*futureIterator).first);
                        ++result;
                    }
                    *futureIterator = std::pair<InputIterator, std::future<bool>>(first, std::move(std::async(std::launch::async, filterFunctor, *first)));
                }
            }
        }
        while (futures.size() > 0)
        {
            for (auto futureIterator = futures.begin(); futureIterator != futures.end(); ++futureIterator)
            {
                if ((*futureIterator).second.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
                {
                    if ((*futureIterator).second.get() == true)
                    {
                        *result = *((*futureIterator).first);
                        ++result;
                    }
                    futureIterator = futures.erase(futureIterator);
                }
            }
        }
    }
    return result;
}

} /* namespace concurrent */


#endif /* CONCURRENT_HPP */
