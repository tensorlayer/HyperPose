#pragma once
#include <cstdint>
#include <string>

#include <ttl/tensor>

// input an image and resize it to target size
void input_image(const std::string &filename,
                 const ttl::tensor_ref<uint8_t, 3> hwc_buffer,  // [h, w, 3]
                 const ttl::tensor_ref<float, 3> chw_buffer,    // [3, h, w]
                 bool flip_rgb);
