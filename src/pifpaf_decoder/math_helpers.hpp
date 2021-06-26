#pragma once

// x[i] = v
void vfill(float* x, unsigned long n, float v);

// c[i] = a[i] * b[i]
void vmul(const float* a, const float* b, float* c, unsigned long n);

// c[i] = a[i] * b
void vsmul(const float* a, float b, float* c, unsigned long n);
