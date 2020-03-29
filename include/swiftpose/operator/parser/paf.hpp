#pragma once

#include "../../utility/data.hpp"

namespace swiftpose
{

namespace parser
{

class paf {
public:
    paf(int feature_height, int feature_width, int original_height, int original_witdh,
                       int n_joins = 1 + COCO_N_PAIRS /* 1 + COCO_N_PARTS */,
                       int n_connections = COCO_N_PAIRS /* COCO_N_PAIRS */,
                       int gauss_kernel_size = 0);
    auto/* TODO */ process(internal_t& feature_maps) {
        if (feature_maps.size() != 2) {

        }
    }
};

}

}  // namespace swiftpose