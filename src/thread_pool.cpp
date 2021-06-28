#include <hyperpose/utility/thread_pool.hpp>

#if __linux__
#include <cstdio>
#include <iostream>
#include <sched.h>
#endif

namespace hyperpose {

// Implementation:
simple_thread_pool::simple_thread_pool(std::size_t sz)
    : m_shared_src(std::make_shared<pool_src>())
{
    const auto n_physical_cores = std::thread::hardware_concurrency();
    for (int i = 0; i < sz; ++i) {
        auto curr_thread = std::thread(
            [this](std::shared_ptr<pool_src> ptr) {
                while (true) {
                    std::function<void()> task;
                    // >>> Critical region => Begin
                    {
                        std::unique_lock<std::mutex> lock(ptr->queue_mu);
                        ptr->cv.wait(lock, [&] {
                            return ptr->shutdown || !ptr->queue.empty();
                        });
                        if (ptr->shutdown && ptr->queue.empty())
                            return; // Conditions to let the thread go.
                        task = std::move(ptr->queue.front());
                        ptr->queue.pop();
                    }
                    // >>> Critical region => End
                    task();
                    if (ptr->to_finish.fetch_add(-1, std::memory_order_relaxed) == 1)
                        ptr->wait_cv.notify_one();
                }
            },
            m_shared_src);
#if __linux__
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i % n_physical_cores, &cpuset);
        int rc = pthread_setaffinity_np(curr_thread.native_handle(),
            sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
        }
#endif
        curr_thread.detach();
    }
}

void simple_thread_pool::wait()
{
    std::unique_lock lock{ m_shared_src->wait_mu };
    m_shared_src->wait_cv.wait(lock, [&] {
        return m_shared_src->to_finish.load(std::memory_order_relaxed) == 0;
    });
}

simple_thread_pool::~simple_thread_pool()
{
    m_shared_src->shutdown = true;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    m_shared_src->cv.notify_all();
}

} // namespace hyperpose
