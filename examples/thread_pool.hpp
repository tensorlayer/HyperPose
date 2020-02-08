//
// Created by ganler-Mac on 2020-02-07.
//

#pragma once

#include <cstddef>
#include <thread>
#include <future>
#include <array>
#include <queue>
#include <vector>
#include <mutex>
#include <memory>
#include <atomic>
#include <algorithm>
#include <functional>
#include <condition_variable>

class simple_thread_pool final
{ // Simple thread-safe & container-free thread pool.
public:
    const std::size_t size{0};
    explicit simple_thread_pool(std::size_t /* suggested size */ = std::thread::hardware_concurrency() + 2);
    ~simple_thread_pool();
    template <typename Func, typename ... Args>
    auto enqueue(Func &&f, Args &&... args) /* For Cpp14+ -> decltype(auto). */
    -> std::future<typename std::result_of<Func(Args...)>::type>;
//    template <typename Func, typename ... Args>
//    auto enqueue_advance(Func &&f, Args &&... args) /* For Cpp14+ -> decltype(auto). */
//    -> std::future<typename std::result_of<Func(Args...)>::type>;
    void wait();
private:
    using task_type = std::function<void()>;
// Use xx::function<> wrapper is not zero overhead.(See the link below)
// https://www.boost.org/doc/libs/1_45_0/doc/html/function/misc.html#id1285061
// https://www.boost.org/doc/libs/1_45_0/doc/html/function/faq.html#id1284915
    struct pool_src
    {
        std::mutex                             queue_mu;
        std::condition_variable                cv;
        std::queue<task_type>                  queue;
        std::atomic<std::size_t>               n_working {0};
        std::condition_variable                wait_cv;
        bool                                   shutdown    {false};
    };
    std::shared_ptr<pool_src>                  m_shared_src;
};

template <typename Type, typename Func, typename ... Args>
static void try_allocate(Type& task, Func&& f, Args&& ... args)
{
    try{
        task = new typename std::remove_pointer<Type>::type(std::bind(std::forward<Func>(f), std::forward<Args>(args)...));
    } catch (const std::exception& e) {
        if(task != nullptr)
            delete task;
        throw e;
    }
}

template <typename Func, typename ... Args>
auto simple_thread_pool::enqueue(Func &&f, Args &&... args) -> std::future<typename std::result_of<Func(Args...)>::type>
{
    using return_type = typename std::result_of<Func(Args...)>::type;
    std::packaged_task<return_type()>* task = nullptr;
    try_allocate(task, std::forward<Func>(f), std::forward<Args>(args)...);
    auto result = task->get_future();
    {
        // Critical region.
        std::lock_guard<std::mutex> lock(m_shared_src->queue_mu);
        m_shared_src->queue.push(
                [task]()
                {
                    (*task)();
                    delete task;
                });
    }
    m_shared_src->cv.notify_one();
    return result;
}


using thread_pool = simple_thread_pool;