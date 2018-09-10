#include <array>
#include <cassert>
#include <cmath>

#include "tensor.h"
namespace
{
constexpr int ksize = 9;

constexpr std::array<float, ksize> gauss_kernal_1d =
    // output of ./norm-cdf.py
    {{
        0.003846046744502994,
        0.02567250067655376,
        0.09944673063266776,
        0.22394038885891837,
        0.2934869017717783,
        0.22394038885891854,
        0.09944673063266762,
        0.02567250067655369,
        0.003846046744503062,
    }};
}  // namespace

const auto gauss_kernel = []() {
    tensor_t<float, 2> x(nullptr, ksize, ksize);

    float sum = 0;
    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) {
            const float v = std::sqrt(gauss_kernal_1d[i] * gauss_kernal_1d[j]);
            sum += v;
            x.at(i, j) = v;
        }
    }

    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) { x.at(i, j) /= sum; }
    }

    return x;
}();
