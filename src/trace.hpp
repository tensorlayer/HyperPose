#pragma once

#ifdef ENABLE_TRACE

// #include <tracer/simple>
#include <tracer/simple_log>

#else

#define TRACE_SCOPE(name)
#define LOG_SCOPE(name)
#define TRACE_STMT(e) e;
#define TRACE_EXPR(e) e
#define DEFINE_TRACE_CONTEXTS

#endif
