#include <hyperpose/utility/thread_pool.hpp>

namespace hyperpose {

// Implementation:
simple_thread_pool::simple_thread_pool(std::size_t sz)
    : m_shared_src(std::make_shared<pool_src>())
{
    for (int i = 0; i < sz; ++i) {
        std::thread(
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
            m_shared_src)
            .detach();
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
