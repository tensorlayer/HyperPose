#pragma once

#include <chrono>
#include <cstdio>
#include <string>

template <typename clock_t> class tracer_t_
{
  public:
    tracer_t_(const std::string &name) : name_(name), t0_(clock_t::now())
    {
        fprintf(stderr, "%s started\n", name_.c_str());
    }

    ~tracer_t_()
    {
        const auto now = clock_t::now();
        const std::chrono::duration<double> d = now - t0_;
        fprintf(stderr, "%s took %.6fs\n", name_.c_str(), d.count());
    }

  private:
    const std::string name_;
    const std::chrono::time_point<clock_t> t0_;
};

using tracer_t = tracer_t_<std::chrono::high_resolution_clock>;
