#pragma once

#include <utility>
#include <vector>

inline bool is_virtual_pair(int pair_id) { return pair_id > 16; }
using idx_pair_t = std::pair<int, int>;
using coco_pair_list_t = std::vector<idx_pair_t>;

inline const coco_pair_list_t COCOPAIRS_NET = {
    { 12, 13 }, // 6
    { 20, 21 }, // 10
    { 14, 15 }, // 7
    { 16, 17 }, // 8
    { 22, 23 }, // 11
    { 24, 25 }, // 12
    { 0, 1 }, // 0
    { 2, 3 }, // 1
    { 4, 5 }, // 2
    { 6, 7 }, // 3
    { 8, 9 }, // 4
    { 10, 11 }, // 5
    { 28, 29 }, // 14
    { 30, 31 }, // 15
    { 34, 35 }, // 17
    { 32, 33 }, // 16
    { 36, 37 }, // 18
    { 18, 19 }, // 9
    { 26, 27 }, // 13
};

inline const coco_pair_list_t COCOPAIRS = {
    { 1, 2 }, // 6
    { 1, 5 }, // 10
    { 2, 3 }, // 7
    { 3, 4 }, // 8
    { 5, 6 }, // 11
    { 6, 7 }, // 12
    { 1, 8 }, // 0
    { 8, 9 }, // 1
    { 9, 10 }, // 2
    { 1, 11 }, // 3
    { 11, 12 }, // 4
    { 12, 13 }, // 5
    { 1, 0 }, // 14
    { 0, 14 }, // 15
    { 14, 16 }, // 17
    { 0, 15 }, // 16
    { 15, 17 }, // 18
    { 2, 16 }, // * 9
    { 5, 17 }, // * 13
};