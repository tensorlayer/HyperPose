#pragma once

#include <mutex>
#include <condition_variable>

namespace swiftpose {

template <typename T>
class thread_safe_queue {
public:
    thread_safe_queue(size_t max_size)
        : m_capacity(max_size), m_array(new T[max_size+1]), m_head(0), m_back(0), m_size(0)
    {}

    ~thread_safe_queue() {
        std::lock_guard lk{m_mu};
        delete m_array;
    }

    void push(T v) {
        std::lock_guard lk{m_mu};
        m_cv.wait_until(lk, [this](){return !unsafe_is_full();});
        m_array[m_back] = std::move(v);
        unsafe_step_back();
        ++m_size;
    }

    template <typename C>
    void push(C container) {
        std::lock_guard lk{m_mu};
        m_cv.wait(lk, [this](){return !unsafe_is_full();});

        if (container.size() > m_capacity - m_size)
            throw std::length_error("Cannot accept so many elements.");

        size_t back2cap = m_capacity - m_back;
        if (back2cap >= container.size())
            std::move(container.begin(), container.end(), m_array + m_head);
        else {
            std::move(std::next(container.begin(), back2cap), container.end(), m_array);
            std::move(container.begin(), std::next(container.begin(), back2cap), m_array + m_head);
        }

        unsafe_step_back(container.size());
        m_size += container.size();
    }

    size_t capacity() const noexcept {
        return m_capacity;
    }

    T dump() {
        T ret{};

        {
            std::lock_guard lk{m_mu};
            ret = std::move(m_array[m_head]);

            unsafe_step_head();
            --m_size;
        }

        m_cv.notify_one();

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

            m_size -= n;
            unsafe_step_head(n);
        }

        for (size_t j = 0; j < n; ++j)
            m_cv.notify_one();

        return ret;
    }

private:
    bool unsafe_is_full() const {
        return m_size == capacity();
    }

    void unsafe_step_back(size_t n = 1) {
        m_back = (m_back + n) % m_capacity;
    }

    void unsafe_step_head(size_t n = 1) {
        m_back = (m_head + n) % m_capacity;
    }

    std::mutex              m_mu;
    std::condition_variable m_cv;

    size_t                  m_head;
    size_t                  m_back;
    size_t                  m_size;
    T*                      m_array;

    const size_t            m_capacity;
};

}