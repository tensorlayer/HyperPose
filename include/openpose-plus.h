#pragma once

#ifdef __cplusplus
#include <openpose-plus.hpp>  // C++ APIs
#include <openpose-plus/human.h>

extern "C" {
#endif

// TODO: move them into COCO
const int n_joins = 18 + 1;
const int n_connections = 17 + 2;

/* C APIs */

// get human from (conf, paf).
// peak tensor will be inferred with default operator.
extern void
process_conf_paf(int height, int width, int n_joins, int n_connections,
                 const float *peaks_,  // [n_joins, height, width]
                 const float *pafmap_  // [2 * n_connections, height, width]
);

#ifdef __cplusplus
}
#endif
