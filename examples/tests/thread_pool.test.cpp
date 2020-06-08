#include <hyperpose/utility/thread_pool.hpp>

#include <iostream>
#include <numeric>
#include <string>

using namespace hyperpose;

// TODO: Change a test case. My recursive parallel qsort seems to be slower than
// single-thread version in "release" mode.
// =============== About test ================
namespace naive {

template <typename Iter>
static Iter partition(const Iter beg, const Iter end)
{
    std::swap(*(end - 1), *(beg + std::rand() % std::distance(beg, end)));
    const auto par = *(end - 1);
    auto base_iter = beg;
    for (auto it = beg; it < end - 1; ++it)
        if (*it < *(end - 1))
            std::swap(*it, *(base_iter++));
    std::swap(*(end - 1), *base_iter);
    return base_iter;
}

template <typename Iter>
static void quick_sort(Iter beg, Iter end)
{
    if (std::distance(beg, end) > 1) {
        auto piv = partition(beg, end);
        quick_sort(beg, piv);
        quick_sort(piv + 1, end);
    }
}

} // namespace naive

namespace pool_v {

template <typename Iter>
static void quick_sort(Iter beg, Iter end, thread_pool& pool)
{
    const auto dis = std::distance(beg, end);
    if (dis > 1) {
        auto piv = ::naive::partition(beg, end);
        if (dis > (1 << 12)) {
            using future_t = std::future<void>;
            auto l = pool.enqueue([beg, piv, &pool]() { quick_sort(beg, piv, pool); });
            quick_sort(piv + 1, end, pool);
        } else {
            naive::quick_sort(beg, piv);
            naive::quick_sort(piv + 1, end);
        }
    }
}

} // namespace pool_v

namespace async_v {

template <typename Iter>
void quick_sort(Iter beg, Iter end)
{
    const auto dis = std::distance(beg, end);
    if (dis > 1) {
        auto piv = ::naive::partition(beg, end);
        if (dis > (1 << 12)) {
            auto foo1 = std::async(std::launch::async,
                [beg, piv]() { quick_sort(beg, piv); });
            auto foo2 = std::async(std::launch::async,
                [piv, end]() { quick_sort(piv + 1, end); });
        } else {
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
        std::cout << "[Bench = end] \t@ " << log << ": \tFor \t<<< " << loop_tms
                  << " >>> times, cost \t<<<"
                  << std::chrono::duration<double, std::milli>(end - beg).count()
                  << ">>> ms" << std::endl;
    }
}

template <typename... Args>
void real_assert(bool result, std::size_t line_number, Args&&... args)
{
    if (!result) {
        std::cerr << "[TEST FAILED] at line " << line_number;
        if constexpr (sizeof...(args) != 0) {
            std::cerr << "\n\t [NOTABLE INFO]\t";
            ((std::cerr << args << "\n") << ...);
        }
    }
};

static void test()
{
    thread_pool pool(std::thread::hardware_concurrency());
    const auto test_once = [&](std::size_t amount) {
        std::vector<int> data(amount);

        auto num_str = std::to_string(amount);
        ::bench(
            [&]() {
                std::generate(data.begin(), data.end(), std::rand);
                ::naive::quick_sort(data.begin(), data.end());
                real_assert(std::is_sorted(data.begin(), data.end()), __LINE__);
            },
            "Single-Thread Qsort Test\t(size = " + num_str + ")", 50);

        ::bench(
            [&]() {
                std::generate(data.begin(), data.end(), std::rand);
                ::async_v::quick_sort(data.begin(), data.end());
                real_assert(std::is_sorted(data.begin(), data.end()), __LINE__);
            },
            "Standard Async Qsort Test\t(size = " + num_str + ")", 50);

        ::bench(
            [&]() {
                std::generate(data.begin(), data.end(), std::rand);
                ::pool_v::quick_sort(data.begin(), data.end(), pool);
                pool.wait();
                real_assert(std::is_sorted(data.begin(), data.end()), __LINE__);
            },
            "Thread-Pool Qsort Test\t(size = " + num_str + ")", 50);
    };

    test_once(1 << 12);
    test_once(1 << 14);
    test_once(1 << 16);
    test_once(1 << 18);
}

int main() { ::test(); }