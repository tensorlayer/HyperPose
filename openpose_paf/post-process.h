#pragma once

#include "tensor.h"

void resize_area(const tensor_t<float, 3> &input, tensor_t<float, 3> &output);
void get_peak(const tensor_t<float, 3> &input, tensor_t<float, 3> &output);
