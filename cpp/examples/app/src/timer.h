#pragma once

#include <chrono>
#include <cstdio>
#include <string>

class timer_t
{
  public:
    timer_t(const std::string &name)
        : name_(name), t0_(std::chrono::system_clock::now())
    {
    }

    ~timer_t()
    {
        const auto now = std::chrono::system_clock::now();
        const std::chrono::duration<double> d = now - t0_;
        fprintf(stderr, "%s took %.2fs\n", name_.c_str(), d.count());
    }

  private:
    const std::string name_;
    const std::chrono::time_point<std::chrono::system_clock> t0_;
};
