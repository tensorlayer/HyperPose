#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace hyperpose {

class simple_thread_pool
    final { // Simple thread-safe & container-free thread pool.
public:
    explicit simple_thread_pool(std::size_t /* suggested size */ = std::thread::hardware_concurrency() + 2);

    ~simple_thread_pool();

    template <typename Func, typename... Args>
    auto enqueue(Func&& f, Args&&... args) /* For Cpp14+ -> decltype(auto). */
        -> std::future<typename std::result_of<Func(Args...)>::type>;

    //    template <typename Func, typename ... Args>
    //    auto enqueue_advance(Func &&f, Args &&... args) /* For Cpp14+ ->
    //    decltype(auto). */
    //    -> std::future<typename std::result_of<Func(Args...)>::type>;
    void wait();

private:
    using task_type = std::function<void()>;
    // Use xx::function<> wrapper is not zero overhead.(See the link below)
    // https://www.boost.org/doc/libs/1_45_0/doc/html/function/misc.html#id1285061
    // https://www.boost.org/doc/libs/1_45_0/doc/html/function/faq.html#id1284915
    struct pool_src {
        std::mutex queue_mu;
        std::mutex wait_mu;
        std::condition_variable cv;
        std::condition_variable wait_cv;
        std::queue<task_type> queue;
        std::atomic<std::size_t> to_finish{ 0 };
        bool shutdown{ false };
    };
    std::shared_ptr<pool_src> m_shared_src;
};

template <typename Func, typename... Args>
std::future<typename std::result_of<Func(Args...)>::type>
simple_thread_pool::enqueue(Func&& f, Args&&... args)
{
    using return_type = typename std::result_of<Func(Args...)>::type;
    using package_t = std::packaged_task<return_type()>;
    package_t* task_ptr = nullptr;

    try {
        task_ptr = new package_t(
            std::bind(std::forward<Func>(f), std::forward<Args>(args)...));
    } catch (const std::exception& e) {
        if (task_ptr != nullptr)
            delete task_ptr;
        throw e;
    }

    auto result = task_ptr->get_future();
    {
        // Critical region.
        std::lock_guard<std::mutex> lock(m_shared_src->queue_mu);
        m_shared_src->queue.push([task_ptr]() {
            (*task_ptr)();
            delete task_ptr;
        });
    }
    m_shared_src->to_finish.fetch_add(1, std::memory_order_relaxed);
    m_shared_src->cv.notify_one();
    return result;
}

using thread_pool = simple_thread_pool;

} // namespace hyperpose