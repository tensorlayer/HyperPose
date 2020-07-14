#pragma once

#include <cassert>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace hyperpose {

template <typename T>
class thread_safe_queue {
public:
    thread_safe_queue(size_t max_size)
        : m_capacity(max_size)
        , m_array(new T[max_size + 1])
        , m_head(0)
        , m_back(0)
        , m_size(0)
    {
    }

    ~thread_safe_queue() { delete[] m_array; }

    void push(T v)
    {
        //        std::cout << "BEGIN: " << __PRETTY_FUNCTION__  << '\n';
        std::lock_guard lk{ m_mu };

        if (m_size >= m_capacity)
            throw std::overflow_error("queue size overflow, max size = " + std::to_string(m_capacity));
        m_array[m_back] = std::move(v);
        unsafe_step_back();
        ++m_size;

        //        std::cout << "END: " << __PRETTY_FUNCTION__  << '\n';
    }

    template <typename Iter>
    void push(Iter begin, Iter end)
    {
        const size_t span_size = std::distance(begin, end);

        if (span_size == 0)
            return;

        if (span_size > m_capacity)
            throw std::logic_error(
                "logic error: `container.size() > m_capacity` is true!");

        std::lock_guard lk{ m_mu };

        if (m_size + span_size > m_capacity)
            throw std::overflow_error("queue size overfloor, max size = " + std::to_string(m_capacity));

        for (auto it = begin; it != end; ++it) {
            m_array[m_back] = *it;
            unsafe_step_back(1);
        }

        m_size += span_size;
    }

    template <typename C>
    void push(C container)
    {
        //        std::cout << "BEGIN: " << __PRETTY_FUNCTION__  << '\n';
        push(std::move_iterator{ container.begin() },
            std::move_iterator{ container.end() });
        //        std::cout << "END: " << __PRETTY_FUNCTION__  << '\n';
    }

    void wait_until_pushed(T t)
    {
        while (true) {
            try {
                push(std::move(t));
            } catch (const std::overflow_error& err) {
                m_wait_for_space += 1;
                std::unique_lock lk{ m_mu };
                m_cv.wait(lk, [this] { return m_size < m_capacity; });
                m_wait_for_space -= 1;
                continue;
            } catch (const std::exception& err) {
                throw;
            }
            break;
        }
    }

    template <typename Iter>
    void wait_until_pushed(Iter begin, Iter end)
    {
        size_t span_size = std::distance(begin, end);
        while (true) {
            try {
                push(begin, end);
            } catch (const std::logic_error& err) {
                //                std::cerr << err.what() << std::endl;
                throw;
            } catch (const std::overflow_error& err) {
                //                std::cerr << err.what() << '\n';
                //                std::cerr << "Currently: size/capacity: " << m_size << '/' << m_capacity << ", space needed: " << span_size << std::endl;
                m_wait_for_space += span_size;
                std::unique_lock lk{ m_mu };
                m_cv.wait(
                    lk, [this, span_size] { return m_capacity - m_size >= span_size; });
                m_wait_for_space -= span_size;
                continue;
            } catch (const std::exception& err) {
                throw;
            }
            break;
        }
    }

    template <typename C>
    void wait_until_pushed(C c)
    {
        wait_until_pushed(std::make_move_iterator(c.begin()),
            std::make_move_iterator(c.end()));
    }

    [[nodiscard]] size_t capacity() const noexcept { return m_capacity; }

    std::optional<T> dump()
    {
        T ret{};

        {
            std::lock_guard lk{ m_mu };

            if (m_size == 0)
                return std::nullopt;

            ret = std::move(m_array[m_head]);

            unsafe_step_head();
            --m_size;

            if (m_wait_for_space != 0)
                m_cv.notify_one();
        }

        return ret;
    }

    std::vector<T> dump(size_t n)
    {
        std::vector<T> ret;
        size_t len = std::min(m_size, n);
        ret.reserve(len);

        {
            std::lock_guard lk{ m_mu };

            for (size_t i = 0; i < len; ++i)
                ret.push_back(std::move(m_array[(m_head + i) % m_capacity]));
            m_size -= len;
            unsafe_step_head(len);
            if (m_wait_for_space != 0)
                m_cv.notify_one();
        }

        return ret;
    }

    std::vector<T> dump_all() { return this->dump(m_capacity); }

    size_t unsafe_size() const { return m_size; }

    friend class basic_stream_manager;

private:
    void unsafe_step_back(size_t n = 1) { m_back = (m_back + n) % m_capacity; }

    void unsafe_step_head(size_t n = 1) { m_head = (m_head + n) % m_capacity; }

    std::mutex m_mu;
    std::condition_variable m_cv;

    size_t m_head;
    size_t m_back;
    size_t m_size;
    T* m_array;
    std::atomic<size_t> m_wait_for_space = 0;

    const size_t m_capacity;
};

} // namespace hyperpose