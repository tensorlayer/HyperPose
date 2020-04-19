#pragma once

#include <mutex>
#include <condition_variable>
#include <vector>
#include <future>
#include <optional>
#include <iostream>
#include <numeric>
#include <cassert>

namespace swiftpose {

template <typename T>
class thread_safe_queue {
public:
    thread_safe_queue(size_t max_size)
        : m_capacity(max_size), m_array(new T[max_size+1]), m_head(0), m_back(0), m_size(0)
    {}

    ~thread_safe_queue() {
        delete[] m_array;
    }

    void push(T v) {
        std::lock_guard lk{m_mu};
        if (m_size + 1 > m_capacity)
            throw std::overflow_error("queue size overfloor!");
        m_array[m_back] = std::move(v);
        unsafe_step_back();
        ++m_size;
    }

    template <typename C>
    void push(C container) {
        std::lock_guard lk{m_mu};

        if (m_size+container.size() > m_capacity)
            throw std::overflow_error("queue size overfloor!");

        size_t back2cap = m_capacity - m_back;
        if (back2cap >= container.size())
            std::move(container.begin(), container.end(), m_array + m_back);
        else {
            std::move(std::next(container.begin(), back2cap), container.end(), m_array);
            std::move(container.begin(), std::next(container.begin(), back2cap), m_array + m_back);
        }

        unsafe_step_back(container.size());
        m_size += container.size();
    }

    [[nodiscard]] size_t capacity() const noexcept {
        return m_capacity;
    }

    std::optional<T> dump() {
        T ret{};

        {
            std::lock_guard lk{m_mu};

            if (m_size == 0)
                return std::nullopt;

            ret = std::move(m_array[m_head]);

            unsafe_step_head();
            --m_size;
        }

        return ret;
    }

    std::vector<T> dump(size_t n) {
        std::vector<T> ret;
        size_t len = std::min(m_size, n);
        ret.reserve(len);

        {
            std::lock_guard lk{m_mu};

            for (size_t i = 0; i < len; ++i)
                ret.push_back(std::move(m_array[(m_head + i) % m_capacity]));

            m_size -= len;
            unsafe_step_head(len);
        }

        return ret;
    }

private:
    void unsafe_step_back(size_t n = 1) {
        m_back = (m_back + n) % m_capacity;
    }

    void unsafe_step_head(size_t n = 1) {
        m_head = (m_head + n) % m_capacity;
    }

    std::mutex              m_mu;

    size_t                  m_head;
    size_t                  m_back;
    size_t                  m_size;
    T*                      m_array;

    const size_t            m_capacity;
};

}

/*
// Test codes.
int main() {
    using namespace swiftpose;

    { // Test 1
        constexpr size_t iters = 10;
        thread_safe_queue<std::vector<int>> queue(iters * 3);
        std::atomic<size_t> answer = 0;

        {
            std::vector<std::future<void>> vec;

            std::vector template1 = {1, 3, 4};
            std::vector template2 = {11, 2, 62, 14, 231};
            std::vector template3 = {13, 534, 12, 5, 31, 2};

            answer += std::accumulate(template1.begin(), template1.end(), 0) * iters;
            answer += std::accumulate(template2.begin(), template2.end(), 0) * iters;
            answer += std::accumulate(template3.begin(), template3.end(), 0) * iters;

            vec.push_back(std::async([&]{queue.push(template1);}));
            vec.push_back(std::async([&]{queue.push(template2);}));
            vec.push_back(std::async([&]{queue.push(template3);}));

            auto f = std::async([&answer, &queue]{
                while(answer != 0) {
                    auto v = queue.dump();
                    if (v.has_value())
                        for(auto&& x : v.value())
                            answer -= x;
                }
            });

            for (size_t i = 0; i < iters - 1; ++i) {
                vec.push_back(std::async([&]{queue.push(template1);}));
                vec.push_back(std::async([&]{queue.push(template2);}));
                vec.push_back(std::async([&]{queue.push(template3);}));
            }

            for(auto&& x : vec)
                x.get();
            f.get();
        }

        assert(answer == 0);
    }

    { // Test 2
        constexpr size_t iters = 10;
        thread_safe_queue<std::vector<int>> queue(iters * 3);
        std::atomic<size_t> answer = 0;

        {
            std::vector<std::future<void>> vec;

            std::vector template1 = {1, 3, 4};
            std::vector template2 = {11, 2, 62, 14, 231};
            std::vector template3 = {13, 534, 12, 5, 31, 2};

            answer += std::accumulate(template1.begin(), template1.end(), 0) * iters;
            answer += std::accumulate(template2.begin(), template2.end(), 0) * iters;
            answer += std::accumulate(template3.begin(), template3.end(), 0) * iters;

            vec.push_back(std::async([&]{queue.push(template1);}));
            vec.push_back(std::async([&]{queue.push(template2);}));
            vec.push_back(std::async([&]{queue.push(template3);}));

            auto f = std::async([&answer, &queue]{
                while(answer != 0) {
                    auto v = queue.dump(queue.capacity());
                    for(auto&& x : v) {
                        for (auto&& y : x)
                            answer -= y;
                    }
                }
            });

            for (size_t i = 0; i < iters - 1; ++i) {
                vec.push_back(std::async([&]{queue.push(template1);}));
                vec.push_back(std::async([&]{queue.push(template2);}));
                vec.push_back(std::async([&]{queue.push(template3);}));
            }

            for(auto&& x : vec)
                x.get();
            f.get();
        }

        assert(answer == 0);
    }
}
*/