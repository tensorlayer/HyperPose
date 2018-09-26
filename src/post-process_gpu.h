#pragma once
#include "tensor.h"

void get_peak_map_gpu(const tensor_t<float, 3> &input,
                      tensor_t<float, 3> &output);
