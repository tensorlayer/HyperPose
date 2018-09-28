#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>

// FIXME: double check
template <typename T> class channel
{
  public:
    channel(int cap) : cap(cap) {}

    T get()
    {
        // std::lock_guard<std::mutex> _(mu);
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&]() { return buffer.size() > 0; });

        const T x = buffer.front();
        buffer.pop();

        // lk.unlock();
        cv.notify_one();

        return x;
    }

    void put(T x)
    {
        // std::lock_guard<std::mutex> _(mu);
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&]() { return buffer.size() < cap; });
        buffer.push(x);

        // lk.unlock();
        cv.notify_one();
    }

  private:
    const int cap;

    std::mutex mu;
    std::queue<T> buffer;

    std::condition_variable cv;
};
