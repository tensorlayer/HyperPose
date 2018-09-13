#pragma once
#include <chrono>
#include <cstdio>
#include <deque>
#include <map>
#include <string>

struct tracer_ctx_t {
    const std::string name;
    const std::chrono::time_point<std::chrono::system_clock> t0;
    int depth;

    using duration_t = std::chrono::duration<double>;
    std::map<std::string, duration_t> total_durations;
    std::map<std::string, uint32_t> call_times;
    std::deque<FILE *> log_files;

    explicit tracer_ctx_t(const std::string &name)
        : name(name), t0(std::chrono::system_clock::now()), depth(0)
    {
        log_files.push_front(stdout);
    }

    ~tracer_ctx_t();

    void in() { ++depth; }

    void out(const std::string &, const duration_t &);

    void indent(FILE * /* fp */ = stdout);

    template <typename... Args> void logf1(FILE *fp, const Args &... args)
    {
        fprintf(fp, "// ");
        fprintf(fp, args...);
        fputc('\n', fp);
    }

    template <typename... Args> void logf(const Args &... args)
    {
        for (auto fp : log_files) {
            if (fp == stdout) { indent(fp); }
            logf1(fp, args...);
            break;  // only log to the first
        }
    }

    void report(FILE *fp) const;
};

struct tracer_t {
    const std::string name;
    const std::chrono::time_point<std::chrono::system_clock> t0;
    tracer_ctx_t &ctx;

    tracer_t(const std::string &, tracer_ctx_t &);
    ~tracer_t();
};

extern tracer_ctx_t default_tracer_ctx;

struct set_trace_log_t {
    const std::string name;
    tracer_ctx_t &ctx;

    set_trace_log_t(const std::string &, bool /* reuse */ = false,
                    tracer_ctx_t & /* ctx */ = default_tracer_ctx);
    ~set_trace_log_t();
};

#define TRACE(name) tracer_t _((name), default_tracer_ctx)

#define _TRACE_WITH_NAMD(name, e)                                              \
    {                                                                          \
        tracer_t _(name, default_tracer_ctx);                                  \
        e;                                                                     \
    }

#define TRACE_IT(e) _TRACE_WITH_NAMD(#e, e);

#define TRACE_NAME(name, e) _TRACE_WITH_NAMD(std::string(#e "::") + name, e);

#define SET_TRACE_LOG(name) set_trace_log_t ___(name)

template <typename... Args> void logf(const Args &... args)
{
    default_tracer_ctx.logf(args...);
}

template <bool enable = false, typename F, typename... Arg>
void trace_call(const std::string &name, F &f, Arg &... args)
{
    if (enable) {
        tracer_t _(name, default_tracer_ctx);
        f(args...);
    } else {
        f(args...);
    }
}
