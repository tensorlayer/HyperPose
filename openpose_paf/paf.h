#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// get human from (conf, paf).
// peak tensor will be inferred with default operator.
extern void
process_conf_paf(int height, int width,  //
                 int channel_j,          // channel_j >= n_joins
                 int channel_c,          // channel_c >= n_connections
                 const float *peaks_,    // [height, width, channel_j]
                 const float *pafmap_    // [height, width, channel_c * 2]
);

// get human from (conf, peak, paf), with user provided peak tensor.
// extern void
// process_conf_peak_paf(int height, int width,
//                       int channel_j,          // channel_j >= n_joins
//                       int channel_c,          // channel_c >= n_connections
//                       const float *heatmap_,  // [height, width, channel_j]
//                       const float *peaks_,    // [height, width, channel_j]
//                       const float *pafmap_    // [height, width, channel_c *
//                       2]
// );

#ifdef __cplusplus
}
#endif
