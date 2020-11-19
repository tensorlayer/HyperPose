
#include "math_helpers.hpp"
#include <cassert>

#ifdef __APPLE__
#define MATH_HELPERS_ACCELERATE 1
#else
#define MATH_HELPERS_ACCELERATE 0
#endif

#if MATH_HELPERS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cmath>
#endif

void vfill(float*x, unsigned long n, float v) {
#if MATH_HELPERS_ACCELERATE
  vDSP_vfill(&v, x, 1, n);
#else
  // Slow version
  for (unsigned long i = 0; i < n; ++i) {
    x[i] = v;
  }
#endif
}

void vadd(const float *a, const float *b, float *c, unsigned long n) {
#if MATH_HELPERS_ACCELERATE
  vDSP_vadd(a, 1, b, 1, c, 1, n);
#else
  // Slow version
  for (unsigned long i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
#endif
}

void vexp(float *x, unsigned long n) {
#if MATH_HELPERS_ACCELERATE
  int n_ = (int)n;
  vvexpf(x, x, &n_);
#else
  // Slow version
  for (unsigned long i = 0; i < n; ++i) {
    x[i] = std::exp(x[i]);
  }
#endif
}

void vmul(const float *a, const float *b, float *c, unsigned long n) {
#if MATH_HELPERS_ACCELERATE
  vDSP_vmul(a, 1, b, 1, c, 1, n);
#else
  // Slow version
  for (unsigned long i = 0; i < n; ++i) {
    c[i] = a[i] * b[i];
  }
#endif
}

void vsmul(const float *a, float b, float *c, unsigned long n) {
#if MATH_HELPERS_ACCELERATE
  vDSP_vsmul(a, 1, &b, c, 1, n);
#else
  // Slow version
  for (unsigned long i = 0; i < n; ++i) {
    c[i] = a[i] * b;
  }
#endif
}

float vargmax(const float *x, unsigned long n, int* i) {
  assert(n > 0);
#if MATH_HELPERS_ACCELERATE
  float maxValue = 0.0f;
  vDSP_Length maxIndex = 0;
  vDSP_maxvi(x, 1, &maxValue, &maxIndex, n);
  *i = (int)maxIndex;
  return maxValue;
#else
  // Slow version
  float maxValue = x[0];
  unsigned long maxIndex = 0;
  for (unsigned long i = 1; i < n; ++i) {
    if (x[i] > maxValue) {
      maxValue = x[i];
      maxIndex = i;
    }
  }
  *i = (int)maxIndex;
  return maxValue;
#endif
}
