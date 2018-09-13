#pragma once
#include <map>
#include <vector>

struct body_part_t {
    int part_idx;
    float x;
    float y;
    float score;
};

template <int J> struct human_t_ {
    body_part_t body_parts[J];
    float score;

#define DEBUG
#ifdef DEBUG
    void print() const
    {
        for (int i = 0; i < J; ++i) {
            const auto body_part = body_parts[i];
            if (body_parts.part_idx >= 0) {
                printf("BodyPart:%d-(%.2f, %.2f) score=%.2f ", i, body_part.x,
                       body_part.y, body_part.score);
            }
        }
        printf("score=%.2f\n", score);
    }
#endif
};

using human_t = human_t_<19>;
