#pragma once

#include <array>
#include <vector>

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
extern void
process_conf_peak_paf(int height, int width,
                      int channel_j,          // channel_j >= n_joins
                      int channel_c,          // channel_c >= n_connections
                      const float *heatmap_,  // [height, width, channel_j]
                      const float *peaks_,    // [height, width, channel_j]
                      const float *pafmap_    // [height, width, channel_c * 2]
);

#ifdef __cplusplus
}
#endif

const float THRESH_VECTOR_SCORE = 0.05;
const int THRESH_VECTOR_CNT1 = 8;
const int THRESH_PART_CNT = 4;
const float THRESH_HUMAN_SCORE = 0.4;

const int STEP_PAF = 10;

template <typename T> T sqr(T x) { return x * x; }

template <typename T> struct point_2d {
    T x;
    T y;

    point_2d<T> operator-(const point_2d<T> &p) const
    {
        return point_2d<T>{x - p.x, y - p.y};
    }

    template <typename S> point_2d<S> cast_to() const
    {
        return point_2d<S>{S(x), S(y)};
    }

    T l2() const { return sqr(x) + sqr(y); }
};

struct Peak {
    point_2d<int> pos;
    float score;
    int id;
};

struct VectorXY {
    float x;
    float y;
};

struct ConnectionCandidate {
    int idx1;
    int idx2;
    float score;
    float etc;
};

inline bool operator>(const ConnectionCandidate &a,
                      const ConnectionCandidate &b)
{
    return a.score > b.score;
}

struct Connection {
    int cid1;
    int cid2;
    float score;
    int peak_id1;
    int peak_id2;
};

struct body_part_ret_t {
    int id;  // id of peak in the list of all peaks
    body_part_ret_t() : id(-1) {}
};

template <int J> struct human_ref_t_ {
    int id;
    body_part_ret_t parts[J];
    float score;
    int n_parts;

    human_ref_t_() : id(-1), score(0), n_parts(0) {}

    bool touches(const std::pair<int, int> &p, const Connection &conn) const
    {
        return parts[p.first].id == conn.cid1 ||
               parts[p.second].id == conn.cid2;
    }
};

using human_ref_t = human_ref_t_<18>;
