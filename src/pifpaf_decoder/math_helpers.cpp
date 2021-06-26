#include "math_helpers.hpp"

void vfill(float* x, unsigned long n, float v)
{
    // Slow version
    for (unsigned long i = 0; i < n; ++i) {
        x[i] = v;
    }
}

void vmul(const float* a, const float* b, float* c, unsigned long n)
{
    // Slow version
    for (unsigned long i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

void vsmul(const float* a, float b, float* c, unsigned long n)
{
    // Slow version
    for (unsigned long i = 0; i < n; ++i) {
        c[i] = a[i] * b;
    }
}
