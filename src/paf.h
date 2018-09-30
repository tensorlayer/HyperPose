#pragma once
#include "human.h"

#ifdef __cplusplus
// C++ APIs

class paf_processor
{
  public:
    virtual std::vector<human_t> operator()(const float * /* heatmap */,
                                            const float * /* PAF */,
                                            bool /* use GPU */) = 0;

    virtual ~paf_processor() {}
};

paf_processor *create_paf_processor(int input_height, int input_width,
                                    int height, int width, int n_joins,
                                    int n_connections, int gauss_kernel_size);
#endif

#ifdef __cplusplus
extern "C" {
#endif

namespace
{
const int n_joins = 18 + 1;
const int n_connections = 17 + 2;
}  // namespace

/*
C APIs
*/

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
