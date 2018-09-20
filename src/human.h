#pragma once
#include <cstdio>
#include <map>
#include <vector>

#include "coco.h"

struct body_part_t {
    bool has_value;
    float x;
    float y;
    float score;

    body_part_t() : has_value(false), x(0), y(0), score(0) {}
};

template <int J> struct human_t_ {
    body_part_t parts[J];
    float score;

    void print() const
    {
        for (int i = 0; i < J; ++i) {
            const auto p = parts[i];
            if (p.has_value) {
                printf("BodyPart:%d-(%.2f, %.2f) score=%.2f ", i, p.x, p.y,
                       p.score);
            }
        }
        printf("score=%.2f\n", score);
    }
};

using human_t = human_t_<COCO_N_PARTS>;

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

using human_ref_t = human_ref_t_<COCO_N_PARTS>;
