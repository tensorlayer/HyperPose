#include "thread_pool.hpp"
#include <cassert>

// Implementation:
simple_thread_pool::simple_thread_pool(std::size_t sz)
        : m_shared_src(std::make_shared<pool_src>()), size(sz)
{
    for(int i=0; i<sz; ++i)
    {
        std::thread([this](std::shared_ptr<pool_src> ptr)
                    {
                        while (true)
                        {
                            std::function<void()> task;
                            // >>> Critical region => Begin
                            {
                                std::unique_lock<std::mutex> lock(ptr->queue_mu);
                                ptr->cv.wait(lock, [&] {
                                    return ptr->shutdown or !ptr->queue.empty();
                                });
                                if (ptr->shutdown and ptr->queue.empty())
                                    return; // Conditions to let the thread go.
                                task = std::move(ptr->queue.front());
                                ptr->queue.pop();
                            }
                            // >>> Critical region => End
                            ptr->n_working.fetch_add(1, std::memory_order_release);
                            task();
                            ptr->n_working.fetch_add(-1, std::memory_order_acquire);
                            ptr->wait_cv.notify_one();
                        }
                    }, m_shared_src).detach();
    }
}

#include <iostream>
void simple_thread_pool::wait()
{
    std::unique_lock<std::mutex> lock(m_shared_src->queue_mu);
    m_shared_src->wait_cv.wait(lock, [&] {
        return m_shared_src->queue.empty() and m_shared_src->n_working.load(std::memory_order_relaxed) == 0;
    });

}

simple_thread_pool::~simple_thread_pool()
{
    m_shared_src->shutdown = true;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    m_shared_src->cv.notify_all();
}


// =============== About test ================
#include <iostream>
namespace naive
{

template <typename Iter>
static Iter partition(const Iter beg, const Iter end)
{
    std::swap(*(end-1), *(beg + std::rand() % std::distance(beg, end)));
    const auto par = *(end - 1);
    auto base_iter = beg;
    for(auto it = beg; it < end-1; ++it)
        if(*it < *(end-1))
            std::swap(*it, *(base_iter++));
    std::swap(*(end-1), *base_iter);
    return base_iter;
}

template <typename Iter>
static void quick_sort(Iter beg, Iter end)
{
    if(std::distance(beg, end) > 1)
    {
        auto piv = partition(beg, end);
        quick_sort(beg, piv);
        quick_sort(piv + 1, end);
    }
}

}

namespace pool_v
{

template<typename Iter>
static void quick_sort(Iter beg, Iter end, thread_pool& pool)
{
    const auto dis = std::distance(beg, end);
    if (dis > 1)
    {
        auto piv = ::naive::partition(beg, end);
        if (dis > 2048)
        {
            using future_t = std::future<void>;
            auto l = pool.enqueue([beg, piv, &pool](){
                quick_sort(beg, piv, pool);
            });
            quick_sort(piv + 1, end, pool);
        } else
        {
            naive::quick_sort(beg, piv);
            naive::quick_sort(piv + 1, end);
        }
    }
}

} // namespace pool_v

namespace async_v {

template<typename Iter>
void quick_sort(Iter beg, Iter end) {
    const auto dis = std::distance(beg, end);
    if (dis > 1)
    {
        auto piv = ::naive::partition(beg, end);
        if (dis > 10000)
        {
            auto foo1 = std::async(std::launch::async, [beg, piv]() { quick_sort(beg, piv); });
            auto foo2 = std::async(std::launch::async, [piv, end]() { quick_sort(piv+1, end); });
        }
        else
        {
            naive::quick_sort(beg, piv);
            naive::quick_sort(piv + 1, end);
        }
    }
}

} // namespace async_v

// Test this thread pool using quick sort.

template <typename Func, typename S>
static void bench(Func&& func, const S& log, int loop_tms = 1)
{
    std::cout << "[Bench = beg] \t@ " << log << std::endl;
    {
        auto beg = std::chrono::system_clock::now();
        for (int i = 0; i < loop_tms; ++i) {
            func();
        }
        auto end = std::chrono::system_clock::now();
        std::cout << "[Bench = end] \t@ " << log << ": \tFor \t<<< " << loop_tms << " >>> times, cost \t<<<"
                  << std::chrono::duration<double, std::milli>(end - beg).count() << ">>> ms" << std::endl;
    }
}

static void test()
{
    const auto get_test_data = [](std::size_t test_sz = (1 << 14))
    {
        std::vector<int> random_test(test_sz);
        std::generate(random_test.begin(), random_test.end(), std::rand);
        return random_test;
    };

    thread_pool pool(std::thread::hardware_concurrency() + 2);
    {
        auto&& data_ = get_test_data();

        ::bench([&data_]()
                {
                    auto data = data_;
                    ::naive::quick_sort(data.begin(), data.end());
                    assert(std::is_sorted(data.begin(), data.end()));
                },
                "Single-Thread Qsort Test\t(size = 16384)",
                100
        );

        ::bench([&data_]()
                {
                    auto data = data_;
                    ::async_v::quick_sort(data.begin(), data.end());
                    assert(std::is_sorted(data.begin(), data.end()));
                },
                "Standard Async Qsort Test\t(size = 16384)",
                100
        );

        ::bench([&data_, &pool]()
                {
                    auto data = data_;
                    ::pool_v::quick_sort(data.begin(), data.end(), pool);
                    pool.wait();
                    assert(std::is_sorted(data.begin(), data.end()));
                },
                "Thread-Pool Qsort Test\t(size = 16384)",
                100
        );
    }

    {
        auto&& data_ = get_test_data(65536);

        ::bench([&data_]()
                {
                    auto data = data_;
                    ::naive::quick_sort(data.begin(), data.end());
                    assert(std::is_sorted(data.begin(), data.end()));
                },
                "Single-Thread Qsort Test\t(size = 65536)",
                100
        );

        ::bench([&data_]()
                {
                    auto data = data_;
                    ::async_v::quick_sort(data.begin(), data.end());
                    assert(std::is_sorted(data.begin(), data.end()));
                },
                "Standard Async Qsort Test\t(size = 65536)",
                100
        );

        ::bench([&data_, &pool]()
                {
                    auto data = data_;
                    ::pool_v::quick_sort(data.begin(), data.end(), pool);
                    pool.wait();
                    assert(std::is_sorted(data.begin(), data.end()));
                },
                "Thread-Pool Qsort Test\t(size = 65536)",
                100
        );
    }

    {
        auto&& data_ = get_test_data(262144);

        ::bench([&data_]()
                {
                    auto data = data_;
                    ::naive::quick_sort(data.begin(), data.end());
                    assert(std::is_sorted(data.begin(), data.end()));
                },
                "Single-Thread Qsort Test\t(size = 262144)",
                100
        );

        ::bench([&data_]()
                {
                    auto data = data_;
                    ::async_v::quick_sort(data.begin(), data.end());
                    assert(std::is_sorted(data.begin(), data.end()));
                },
                "Standard Async Qsort Test\t(size = 262144)",
                100
        );

        ::bench([&data_, &pool]()
                {
                    auto data = data_;
                    ::pool_v::quick_sort(data.begin(), data.end(), pool);
                    pool.wait();
                    assert(std::is_sorted(data.begin(), data.end()));
                },
                "Thread-Pool Qsort Test\t(size = 262144)",
                100
        );
    }

}

//int main()
//{
//#ifdef NDEBUG
//    std::cerr << "Please set up debug flag first!\n";
//#else
//    ::test();
//#endif
//}