// Heavily modified from openpifpaf/cpp/example.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <tuple>
#include <utility>

#include "math_helpers.hpp"
#include "openpifpaf_postprocessor.hpp"

struct Occupancy {
    // self.reduction = reduction
    // self.min_scale_reduced = min_scale / reduction
    constexpr static float reduction = 2.f;
    constexpr static float min_scale_reduced = 4.f / reduction;
    size_t d0, d1, d2; // c h w
    std::vector<uint8_t> occupancy_view;

    Occupancy(size_t d0_, size_t d1_, size_t d2_)
        : d0(d0_)
        , d1(d1_)
        , d2(d2_)
        , occupancy_view(d0_ * d1_ * d2_)
    {
    }

    bool fuzz_get(size_t f, float y, float x)
    {
        if (f >= d0)
            return true;

        // scalar_nonzero_clipped_with_reduction
        float xx = std::min((float)d2 - 1, std::max(0.f, x / reduction));
        float yy = std::min((float)d1 - 1, std::max(0.f, y / reduction));

        return get(f, yy, xx);
    }

    bool get(size_t d0_, size_t d1_, size_t d2_)
    {
        return occupancy_view[(d1 * d2) * d0_ + d2 * d1_ + d2_];
    }

    void set(size_t d0_, size_t d1_, size_t d2_)
    {
        occupancy_view[(d1 * d2) * d0_ + d2 * d1_ + d2_] = 1;
    }
};

namespace lpdnn {
namespace aiapp_impl {

    constexpr int OpenPifPafPostprocessor::bones[19][2] = {
        { 16, 14 },
        { 14, 12 },
        { 17, 15 },
        { 15, 13 },
        { 12, 13 },
        { 6, 12 },
        { 7, 13 },
        { 6, 7 },
        { 6, 8 },
        { 7, 9 },
        { 8, 10 },
        { 9, 11 },
        { 2, 3 },
        { 1, 2 },
        { 1, 3 },
        { 2, 4 },
        { 3, 5 },
        { 4, 6 },
        { 5, 7 },
    };

    struct to_point {
        int field_id;
        bool possitve;
    };

    auto BY_SOURCE_MAP = [] {
        // print(self.by_source)
        // for i in range(17):
        //     for (end_i), (caf_i, connect) in self.by_source[i].items():
        //         data = f'to_point{{{caf_i}, {"true" if connect else "false"}}}'
        //         print(f'smap[{i}][{end_i}] = {data};')
        std::array<std::map<int, to_point, std::greater<>>, 17> smap;
        smap[0][1] = to_point{ 13, true };
        smap[0][2] = to_point{ 14, true };
        smap[1][2] = to_point{ 12, true };
        smap[1][0] = to_point{ 13, false };
        smap[1][3] = to_point{ 15, true };
        smap[2][1] = to_point{ 12, false };
        smap[2][0] = to_point{ 14, false };
        smap[2][4] = to_point{ 16, true };
        smap[3][1] = to_point{ 15, false };
        smap[3][5] = to_point{ 17, true };
        smap[4][2] = to_point{ 16, false };
        smap[4][6] = to_point{ 18, true };
        smap[5][11] = to_point{ 5, true };
        smap[5][6] = to_point{ 7, true };
        smap[5][7] = to_point{ 8, true };
        smap[5][3] = to_point{ 17, false };
        smap[6][12] = to_point{ 6, true };
        smap[6][5] = to_point{ 7, false };
        smap[6][8] = to_point{ 9, true };
        smap[6][4] = to_point{ 18, false };
        smap[7][5] = to_point{ 8, false };
        smap[7][9] = to_point{ 10, true };
        smap[8][6] = to_point{ 9, false };
        smap[8][10] = to_point{ 11, true };
        smap[9][7] = to_point{ 10, false };
        smap[10][8] = to_point{ 11, false };
        smap[11][13] = to_point{ 1, false };
        smap[11][12] = to_point{ 4, true };
        smap[11][5] = to_point{ 5, false };
        smap[12][14] = to_point{ 3, false };
        smap[12][11] = to_point{ 4, false };
        smap[12][6] = to_point{ 6, false };
        smap[13][15] = to_point{ 0, false };
        smap[13][11] = to_point{ 1, true };
        smap[14][16] = to_point{ 2, false };
        smap[14][12] = to_point{ 3, true };
        smap[15][13] = to_point{ 0, true };
        smap[16][14] = to_point{ 2, true };
        return smap;
    }();

    static const int C = 17;
    static const float STRIDE = 8.0f;
    static const float seedThreshold = 0.3f; // 0.5
    //static const float keypointThreshold = 0.15f;
    static const float instanceThreshold = 0.2f;

    static void scalarSquareAddConstant(float* field,
        int fieldH,
        int fieldW,
        const std::vector<float>& x,
        const std::vector<float>& y,
        const std::vector<float>& width,
        const std::vector<float>& v)
    {
        // minx_np = np.round(x_np - width_np).astype(np.int)
        // minx_np = np.clip(minx_np, 0, field.shape[1] - 1)
        std::vector<int> minx(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            minx[i] = std::min(fieldW - 1, std::max(0, (int)std::round(x[i] - width[i])));
        }

        // miny_np = np.round(y_np - width_np).astype(np.int)
        // miny_np = np.clip(miny_np, 0, field.shape[0] - 1)
        std::vector<int> miny(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            miny[i] = std::min(fieldH - 1, std::max(0, (int)std::round(y[i] - width[i])));
        }

        // maxx_np = np.round(x_np + width_np).astype(np.int)
        // maxx_np = np.clip(maxx_np + 1, minx_np + 1, field.shape[1])
        std::vector<int> maxx(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            maxx[i] = std::min(fieldW, std::max(minx[i] + 1, (int)std::round(x[i] + width[i]) + 1));
        }

        // maxy_np = np.round(y_np + width_np).astype(np.int)
        // maxy_np = np.clip(maxy_np + 1, miny_np + 1, field.shape[0])
        std::vector<int> maxy(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            maxy[i] = std::min(fieldH, std::max(miny[i] + 1, (int)std::round(y[i] + width[i]) + 1));
        }

        // for i in range(minx.shape[0]):
        //     for xx in range(minx[i], maxx[i]):
        //         for yy in range(miny[i], maxy[i]):
        //             field[yy, xx] += v[i]
        for (size_t i = 0; i < minx.size(); ++i) {
            for (int yy = miny[i]; yy < maxy[i]; ++yy) {
                for (int xx = minx[i]; xx < maxx[i]; ++xx) {
                    field[yy * fieldW + xx] += v[i];
                }
            }
        }
    }

    static void scalarSquareAddGaussWitMax(float* field,
        int fieldH,
        int fieldW,
        const std::vector<float>& x,
        const std::vector<float>& y,
        const std::vector<float>& sigma_,
        const std::vector<float>& v,
        float truncate,
        float max_val = 1.0f)
    {
        // // ganler!
        // assert(v.size() == x.size() == y.size() == sigma_.size());
        for (size_t i = 0; i < x.size(); ++i) {
            float csigma = sigma_[i];
            float truncate_csigma = csigma * truncate;
            float cx = x[i];
            float cy = y[i];
            float cv = v[i];
            const auto clip = [](float val, float low, float high) {
                return std::max(low, std::min(high, val));
            };

            // printf("%f, %f, %f, %f, %f\n", cx, cy, csigma, truncate_csigma, max_val);
            const int64_t minx = clip(cx - truncate_csigma, 0, fieldW - 1);
            const int64_t maxx = clip(cx + truncate_csigma + 1, minx + 1, fieldW);
            const int64_t miny = clip(cy - truncate_csigma, 0, fieldH - 1);
            const int64_t maxy = clip(cy + truncate_csigma + 1, miny + 1, fieldH);
            // std::cout << minx << '\t' << maxx << '\t' << miny << '\t' << maxy << '\n';
            // printf("%lli, %lli, %lli, %lli\n", minx, maxx, miny, maxy);

            for (int64_t xx = minx; xx < maxx; ++xx) {
                float deltax2 = (xx - cx) * (xx - cx);
                for (int64_t yy = miny; yy < maxy; ++yy) {
                    float deltay2 = (yy - cy) * (yy - cy);

                    if (deltax2 + deltay2 > truncate_csigma * truncate_csigma) {
                        continue;
                    }

                    const auto approx_exp = [](float x) {
                        if (x > 2 || x < -2)
                            return 0.f;
                        x = 1.f + x / 8;
                        x *= x;
                        x *= x;
                        x *= x;
                        return x;
                    };
                    float vv = (deltax2 < 0.25 && deltay2 < 0.25) ? cv : cv * approx_exp(-0.5 * (deltax2 + deltay2) / (csigma * csigma));
                    field[yy * fieldW + xx] += vv;
                    field[yy * fieldW + xx] = std::min(max_val, field[yy * fieldW + xx]);
                }
            }
        }
    }

    static void scalarSquareAddSingle(Occupancy& field,
        int field_idx,
        int fieldH,
        int fieldW,
        float x,
        float y,
        float width,
        float reduction = 1.0,
        float min_scaled_reduced = 0.0)
    {
        if (reduction != 1.0) {
            x /= reduction;
            y /= reduction;
            width = std::max(min_scaled_reduced, width / reduction);
        }

        // minx = max(0, int(round(x - width)))
        // miny = max(0, int(round(y - width)))
        auto minx = std::min(fieldW - 1, std::max(0, (int)(x - width)));
        auto miny = std::min(fieldH - 1, std::max(0, (int)(y - width)));

        // maxx = max(minx + 1, min(field.shape[1], int(round(x + width)) + 1))
        // maxy = max(miny + 1, min(field.shape[0], int(round(y + width)) + 1))
        auto maxx = std::min(fieldW, std::max(minx + 1, std::min(fieldW, (int)(x + width) + 1)));
        auto maxy = std::min(fieldH, std::max(miny + 1, std::min(fieldH, (int)(y + width) + 1)));

        // field[miny:maxy, minx:maxx] += value
        for (auto yy = miny; yy < maxy; ++yy) {
            for (auto xx = minx; xx < maxx; ++xx) {
                field.set(field_idx, yy, xx);
            }
        }
    }

    OpenPifPafPostprocessor::Target_intensity
    OpenPifPafPostprocessor::targetIntensities(const std::vector<float>& pif,
        float v_th, bool coreOnly)
    {
        constexpr float PIF_NN = 16.0f;

        const size_t targets_stride_0 = H_hr * W_hr;
        const size_t scales_stride_0 = H_hr * W_hr;
        const size_t ns_stride_0 = H_hr * W_hr;

        // These tensors need to be emptied out on each frame.
        vfill(targetsCoreOnly.data(), targetsCoreOnly.size(), 0.0f);
        vfill(targets.data(), targets.size(), 0.0f);
        vfill(scales.data(), scales.size(), 0.0f);
        vfill(ns.data(), ns.size(), 0.0f);

        std::vector<float> v;
        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> s;

        for (int i = 0; i < C; ++i) {
            // Threshold pif[i, ...], which is a (4, h, w) tensor. Copy the values
            // that are over the threshold into four vectors: v, x, y, s. Multiply
            // x, y, s with the stride.
            //
            // v, x, y, s = p[:, p[0] > v_th]
            // x = x * self.stride
            // y = y * self.stride
            // s = s * self.stride
            v.clear();
            x.clear();
            y.clear();
            s.clear();
            const size_t pifOffset = i * pif_stride_0;
            const size_t xOffset = pifOffset + pif_stride_1;
            const size_t yOffset = xOffset + pif_stride_1;
            const size_t sOffset = yOffset + pif_stride_1 * 2;
            for (int j = 0; j < H * W; ++j) {
                float p = pif[pifOffset + j];
                if (p > v_th) {
                    v.push_back(p);
                    x.push_back(pif[xOffset + j] * STRIDE);
                    y.push_back(pif[yOffset + j] * STRIDE);
                    s.push_back(std::max(1., 0.5 * pif[sOffset + j] * STRIDE));
                }
            }

            /*
    // For debugging
    printf("iteration: %d\n", i);
    printf("v:\n"); for (auto n : v) printf("%f, ", n); printf("\n");
    printf("x:\n"); for (auto n : x) printf("%f, ", n); printf("\n");
    printf("y:\n"); for (auto n : y) printf("%f, ", n); printf("\n");
    printf("s:\n"); for (auto n : s) printf("%f, ", n); printf("\n");
    */

            // Create a high-resolution confidence map for this keypoint.
            // std::cout << x.size() << '\t'<< y.size() << '\t'<< v.size() << '\t' << s.size() << '\n';
            // v / pif_nn
            std::vector<float> v_over_pif_nn(v.size());
            vsmul(v.data(), 1.0f / PIF_NN, v_over_pif_nn.data(), v.size());

            // The original code computed the "core only" version in a separate step
            // but that duplicates a bunch of work, so we do it at the same time.
            const auto tco = targetsCoreOnly.data() + i * targets_stride_0;
            scalarSquareAddGaussWitMax(tco, H_hr, W_hr, x, y, s, v_over_pif_nn, 1.0f, 1.0f);

            size_t cnt = 0;
            for (size_t dd = 0; dd < targets_stride_0; ++dd) {
                if (tco[dd] > 0.01)
                    ++cnt;
            }
            // std::cout << targets_stride_0 << '\t' << i << '\t'<< cnt << '\t' << tco[0] << '\n';

            // s * v
            std::vector<float> s_times_v(v.size());
            vmul(s.data(), v.data(), s_times_v.data(), v.size());

            const auto t = targets.data() + i * targets_stride_0;
            const auto scale = scales.data() + i * scales_stride_0;
            const auto n = ns.data() + i * ns_stride_0;
            scalarSquareAddGaussWitMax(t, H_hr, W_hr, x, y, s, v_over_pif_nn, 1.0f);
            scalarSquareAddConstant(scale, H_hr, W_hr, x, y, s, s_times_v);
            scalarSquareAddConstant(n, H_hr, W_hr, x, y, s, v);
        }

        // m = ns > 0
        // scales[m] = scales[m] / ns[m]
        for (size_t i = 0; i < scales.size(); ++i) {
            const auto d = ns[i];
            if (d > 0) {
                scales[i] /= d;
            }
        }
        return Target_intensity{ targets, scales, targetsCoreOnly };
    }

    std::tuple<float, float, float, float>
    OpenPifPafPostprocessor::growConnectionBlend(float x, float y, float s, const std::array<std::vector<float>, 9>& paf_field)
    {
        // # source value
        // paf_field = paf_center(paf_field, xy[0], xy[1], sigma=2.0)
        // if paf_field.shape[1] == 0:
        //     return 0, 0, 0
        const float sigma = 2.0 * s;
        const float sigma2 = 0.25 * s * s;
        size_t score_1_i = 0, score_2_i = 0;
        float score_1 = 0, score_2 = 0;

        const int paf_stride = paf_field.front().size();
        for (int i = 0; i < paf_stride; ++i) {
            if ((paf_field[1][i] < x - sigma) || (paf_field[1][i] > x + sigma) || (paf_field[2][i] < y - sigma) || (paf_field[2][i] > y + sigma))
                continue;
            float d2 = (paf_field[1][i] - x) * (paf_field[1][i] - x) + (paf_field[2][i] - y) * (paf_field[2][i] - y);
            float score = std::exp(-0.5 * d2 / sigma2) * paf_field[0][i];
            if (score >= score_1) {
                score_2_i = score_1_i;
                score_2 = score_1;
                score_1_i = i;
                score_1 = score;
            } else if (score > score_2) {
                score_2_i = i;
                score_2 = score;
            }
        }

        if (score_1 == 0)
            return { 0, 0, 0, 0 };

        auto entry_1 = std::make_tuple(paf_field[3][score_1_i], paf_field[4][score_1_i], paf_field[6][score_1_i], paf_field[8][score_1_i]);

        auto [ex1, ey1, eb1, es1] = entry_1;
        if (score_2 < 0.01 || score_2 < 0.5 * score_1) {
            return { ex1, ey1, es1, score_1 * 0.5 };
        }

        // blend...
        auto entry_2 = std::make_tuple(paf_field[3][score_2_i], paf_field[4][score_2_i], paf_field[6][score_2_i], paf_field[8][score_2_i]);
        auto [ex2, ey2, eb2, es2] = entry_2;

        float blend_d2 = (ex1 - ex2) * (ex1 - ex2) + (ey1 - ey2) * (ey1 - ey2);
        if (blend_d2 > ((es1 * es1) / 4)) {
            return { ex1, ey1, es1, score_1 * 0.5 };
        }

        return {
            // xysv
            (score_1 * ex1 + score_2 * ex2) / (score_1 + score_2),
            (score_1 * ey1 + score_2 * ey2) / (score_1 + score_2),
            (score_1 * es1 + score_2 * es2) / (score_1 + score_2),
            0.5 * (score_1 + score_2),
        };
    }

    using xysv = std::optional<std::tuple<float, float, float, float>>;

    struct queue_item { // -score, xyv, start_i, end_i
        explicit queue_item(float f, xysv xysv_, int s, int e)
            : data(std::make_tuple(f, std::move(xysv_), s, e))
        {
        }
        std::tuple<float, xysv, int, int> data;
        friend bool operator>(const queue_item& l, const queue_item& r)
        {
            return std::get<0>(l.data) >= std::get<0>(r.data);
        }
        friend bool operator<(const queue_item& l, const queue_item& r)
        {
            return std::get<0>(l.data) < std::get<0>(r.data);
        }
    };

    void OpenPifPafPostprocessor::grow(Annotation& ann,
        const FBContainer& pafForward,
        const FBContainer& pafBackward)
    {
        // frontierActive = true;
        // blockFrontier.clear();
        std::set<std::pair<int, int>> in_frontier{};
        std::priority_queue<queue_item, std::deque<queue_item>, std::greater<>> frontier;

        const auto add_to_frontier = [&](size_t start_i) {
            for (const auto& [end_i, to_p] : BY_SOURCE_MAP[start_i]) {
                int caf_i = to_p.field_id;
                // std::cout << "----> " << start_i << '\t' << end_i << '\t' << caf_i << '\n';
                if (ann.keypoints[3 * end_i + 2] > 0.0) {
                    // std::cout << "CONTINUE start_i = " << start_i << '\n';
                    continue;
                }
                // found!
                if (in_frontier.cend() != in_frontier.find(std::make_pair(start_i, end_i))) {
                    // std::cout << "CONTINUE map already got you!\n";
                    continue;
                }

                float max_possible_score = std::sqrt(ann.keypoints[3 * start_i + 2]);
                // std::cout << "put " << start_i << ' ' << end_i << "\tscore = " << max_possible_score << "\n";
                frontier.emplace(-max_possible_score, std::nullopt, start_i, end_i);
                in_frontier.emplace(start_i, end_i);
            }
        };

        const auto frontier_get = [&]() -> std::optional<queue_item> {
            while (!frontier.empty()) {
                auto entry = frontier.top();
                frontier.pop();

                {
                    auto [_a, _b, start_i, end_i] = entry.data;
                    // std::cout << "POP " << start_i << ' ' << end_i << " has val = " << std::get<1>(entry.data).has_value() << '\n';
                }

                if (std::get<1>(entry.data).has_value()) {
                    // std::cout << "RETURN \n";
                    return entry;
                }

                auto [_a, _b, start_i, end_i] = entry.data;
                if (ann.keypoints[end_i * 3 + 2] > 0.0)
                    continue;

                // connection_value(self, ann, caf_scored, start_i, end_i, *, reverse_match=True):
                auto new_xysv = [&](size_t start_i, size_t end_i) -> xysv {
                    const auto& point = BY_SOURCE_MAP[start_i][end_i];
                    int caf_i = point.field_id;
                    bool is_forward = point.possitve;
                    const auto& caf_f = is_forward ? pafForward[caf_i] : pafBackward[caf_i]; // [19, 9, N]
                    const auto& caf_b = is_forward ? pafBackward[caf_i] : pafForward[caf_i];
                    auto [x, y, v] = std::make_tuple(ann.keypoints[start_i * 3], ann.keypoints[start_i * 3 + 1], ann.keypoints[start_i * 3 + 2]);
                    float xy_scale_s = std::max(0.f, ann.jointScales[start_i]);
                    const auto [nx, ny, ns, nv] = growConnectionBlend(x, y, xy_scale_s, caf_f);
                    // std::cout << "NEW:\t" << nx << '\t' << ny << '\t' << ns << '\t' << nv << '\n';

                    if (nv == 0)
                        return std::nullopt;

                    float keypoint_score = std::sqrt(nv * v);
                    if (keypoint_score < keypointThreshold)
                        return std::nullopt;
                    // Use relative threashold
                    constexpr float keypoint_threshold_rel = 0.5;
                    if (keypoint_score < v * keypoint_threshold_rel)
                        return std::nullopt;

                    float xy_scale_t = std::max(0.f, ns);
                    // if self.reverse_match and reverse_match -> true
                    const auto [rx, ry, rs, rv] = growConnectionBlend(nx, ny, xy_scale_t, caf_b);
                    // std::cout << "REVERSE:\t" << rx << '\t' << ry << '\t' << rs << '\t' << rv << '\n';
                    if (rs == 0 || std::abs(x - rx) + std::abs(y - ry) > xy_scale_s)
                        return std::nullopt;

                    return std::make_tuple(nx, ny, ns, keypoint_score);
                }(start_i, end_i);

                if (std::nullopt == new_xysv)
                    continue;

                frontier.emplace(-std::get<3>(new_xysv.value()), new_xysv, start_i, end_i);
            }
            return std::nullopt;
        };

        for (size_t joint_i = 0; joint_i < N_PIFPAF_KEYPOINTS; ++joint_i) {
            if (ann.keypoints[3 * joint_i + 2] != 0.0) {
                // std::cout << "-----joint_i " << joint_i << "\n";
                add_to_frontier(joint_i);
            }
        }

        while (true) {
            auto entry = frontier_get();
            if (!entry.has_value())
                break;

            auto [_, new_xysv, jsi, jti] = entry.value().data;

            // std::cout << "jsi = " << jsi << ", jti = " << jti << ", ann.data[jti, 2] = " << ann.keypoints[jti * 3 + 2] << '\n';
            if (ann.keypoints[jti * 3 + 2] > 0.0)
                continue;

            auto [nx, ny, ns, nv] = new_xysv.value();
            ann.keypoints[jti * 3 + 0] = nx;
            ann.keypoints[jti * 3 + 1] = ny;
            ann.keypoints[jti * 3 + 2] = nv;
            ann.jointScales[jti] = ns;
            add_to_frontier(jti);
        }
    }

    std::vector<OpenPifPafPostprocessor::Annotation> OpenPifPafPostprocessor::softNMS(std::vector<Annotation>& annotations)
    {
        float maxx = 0.0f;
        float maxy = 0.0f;
        for (auto& ann : annotations) {
            for (int k = 0; k < N_PIFPAF_KEYPOINTS; ++k) {
                auto x = ann.keypoints[k * 3];
                auto y = ann.keypoints[k * 3 + 1];
                if (x > maxx) {
                    maxx = x;
                }
                if (y > maxy) {
                    maxy = y;
                }
            }
        }

        const auto h = (int)(maxy + 1);
        const auto w = (int)(maxx + 1);
        Occupancy occupied(17, h, w);

        std::vector<int> sorted(annotations.size());
        std::iota(sorted.begin(), sorted.end(), 0);
        std::sort(sorted.begin(), sorted.end(), [annotations](int const& a, int const& b) {
            return annotations[a].score() > annotations[b].score();
        });

        for (auto a : sorted) {
            Annotation& ann = annotations[a];
            for (int k = 0; k < N_PIFPAF_KEYPOINTS; ++k) {
                const auto x = ann.keypoints[k * 3];
                const auto y = ann.keypoints[k * 3 + 1];
                const auto v = ann.keypoints[k * 3 + 2];
                if (v == 0) {
                    continue;
                }

                const auto i = std::min(std::max(0, (int)std::round(x)), w - 1);
                const auto j = std::min(std::max(0, (int)std::round(y)), h - 1);

                if (occupied.fuzz_get(k, j, i)) {
                    ann.keypoints[k * 3 + 2] = 0.0f;
                } else {
                    scalarSquareAddSingle(occupied, k, h, w, x, y, ann.jointScales[k]);
                }
            }
        }

        std::vector<Annotation> filtered;
        for (auto& ann : annotations) {
            for (int k = 0; k < N_PIFPAF_KEYPOINTS; ++k) {
                if (ann.keypoints[k * 3 + 2] > 0.0f) {
                    filtered.push_back(ann);
                    break;
                }
            }
        }
        return filtered;

        // Note: The original code sorts here on the score (descending), but
        // we sort again later on so it's a bit quicker if we skip that here.
    }

    void OpenPifPafPostprocessor::initTensors(int tensorWidth, int tensorHeight)
    {
        H = tensorHeight;
        W = tensorWidth;
        H_hr = (H - 1) * (int)STRIDE + 1;
        W_hr = (W - 1) * (int)STRIDE + 1;

        pif_stride_1 = H * W;
        pif_stride_0 = 5 * pif_stride_1;

        pifhr_stride_1 = W_hr;
        pifhr_stride_0 = H_hr * pifhr_stride_1;

        const int shape = C * H_hr * W_hr;
        targetsCoreOnly = std::vector<float>(shape);
        targets = std::vector<float>(shape);
        scales = std::vector<float>(shape);
        ns = std::vector<float>(shape);
    }

    ai_app::Object_detection::Result OpenPifPafPostprocessor::postprocess(
        int inputWidth, int inputHeight,
        int tensorWidth, int tensorHeight,
        const std::vector<float>& pif,
        const std::vector<float>& paf)
    {
        // Allocate the intermediate tensors the first time or when the size changes.
        if (W != tensorWidth || H != tensorHeight) {
            initTensors(tensorWidth, tensorHeight);
        }

        const auto result_tuple = targetIntensities(pif);
        const auto& pifhr = std::get<0>(result_tuple);
        const auto& pifhrScales = std::get<1>(result_tuple);
        const auto& pifhrCore = std::get<2>(result_tuple);

        //      (17, 5, H, W)
        // pif: [v, x, y, _, s]
        const size_t pif_ch = 5, hw_size = H * W;
        const size_t pif_shard_size = pif_ch * hw_size;

        // BEGIN: seeds = utils.CifSeeds(cifhr.accumulated).fill(fields, self.cif_metas)
        std::vector<std::tuple<float, int, float, float, float>> seeds{};

        const float maxx = W_hr - 0.51, maxy = H_hr - 0.51;
        for (size_t field_i = 0; field_i < N_PIFPAF_KEYPOINTS; ++field_i) {
            // Search qualified entries.
            size_t this_field_offset = field_i * pif_shard_size;
            for (size_t hw_index = 0; hw_index < hw_size; ++hw_index) {
                size_t vindex = hw_index + this_field_offset;
                if (pif[vindex] > seedThreshold) {
                    float c = pif[vindex], x = pif[vindex + hw_size], y = pif[vindex + 2 * hw_size], s = pif[vindex + 4 * hw_size];
                    // scalar_values
                    if (x < -0.49 || y < -0.49 || x > maxx || y > maxy) {
                        continue;
                    }
                    float v = pifhrCore[field_i * W_hr * H_hr + ((size_t)(y * STRIDE + 0.5) * W_hr) + (size_t)(x * STRIDE + 0.5)];
                    // scalar_values :: over.

                    v = 0.9 * v + 0.1 * c;
                    // printf("%f   %f, %f, %f, %f\n", v, c, x, y, s);

                    // pass or not?
                    if (v > seedThreshold) {
                        // ok, you pass. -> seeds -> [x, y, v, s]
                        seeds.emplace_back(v, field_i, x * STRIDE, y * STRIDE, s * STRIDE);
                    }
                }
            }
        }
        // std::cout << seeds.size() << "seeds size\n";
        // END: seeds = utils.CifSeeds(cifhr.accumulated).fill(fields, self.cif_metas)

        // BEGIN: caf_scored = utils.CafScored(cifhr.accumulated).fill(fields, self.caf_metas)
        // (19, 9, DYNAMICs)
        constexpr size_t paf_ch = 9;
        const size_t paf_shard_size = paf_ch * hw_size;
        // (19, 9, H, W)...
        FBContainer forward{}, backward{};
        for (size_t field_i = 0; field_i < forward.size(); ++field_i) {
            constexpr float PAF_SCORE_THRE = 0.2;
            constexpr float CIF_FLOOR = 0.1;
            // filter!
            for (size_t hw_idx = 0; hw_idx < hw_size; ++hw_idx) {
                const size_t paf_conf_idx = hw_idx + field_i * paf_shard_size;
                const auto conf = paf[paf_conf_idx];
                if (conf > PAF_SCORE_THRE) {
                    // values in this line...
                    std::array<float, 9> this_ch{};
                    for (size_t chidx = 0; chidx < this_ch.size(); ++chidx) {
                        this_ch[chidx] = paf[paf_conf_idx + chidx * hw_size];
                        if (chidx != 0)
                            this_ch[chidx] *= STRIDE;
                    }

                    auto backward_pif_ch = bones[field_i][0] - 1;
                    auto forward_pif_ch = bones[field_i][1] - 1;
                    // backward pass.
                    constexpr std::array<size_t, 9> BACKWARD_IDX{ 0, 3, 4, 1, 2, 6, 5, 8, 7 };
                    constexpr std::array<size_t, 9> FORWARD_IDX{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };

                    // restore... (yet another filtering...)
                    // cifhr_t = scalar_values(self.cifhr[joint_t], nine[3], nine[4], default=0.0)
                    // nine[0] = nine[0] * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_t)
                    const auto pass = [&this_ch, maxx, maxy, this, field_i, &pifhrCore](const auto& idx_mapping, FBContainer& cont, size_t pif_field_idx) {
                        float x = this_ch[idx_mapping[3]], y = this_ch[idx_mapping[4]];
                        if (!(x < -0.49 || y < -0.49 || x > maxx || y > maxy)) {
                            // std::cout << field_i << "\tXY = \t"<< x << '\t' << y << '\t' << (size_t)(x + 0.5) << '\t' << (size_t)(y + 0.5) << "\t MAX HW: " << W_hr << ' ' << H_hr << std::endl;
                            float cifhr_t = pifhrCore[pif_field_idx * W_hr * H_hr + ((size_t)(y + 0.5) * W_hr) + (size_t)(x + 0.5)];
                            float new_v = this_ch[0] * (CIF_FLOOR + (1 - CIF_FLOOR) * cifhr_t);
                            if (new_v > PAF_SCORE_THRE) {
                                // forward pass.
                                for (size_t fwd_idx = 0; fwd_idx < cont.front().size(); ++fwd_idx) {
                                    // restore!
                                    cont[field_i][fwd_idx].push_back(this_ch[idx_mapping[fwd_idx]]);
                                }
                                cont[field_i][0].back() = new_v;
                            }
                        }
                    };

                    pass(BACKWARD_IDX, backward, backward_pif_ch);
                    pass(FORWARD_IDX, forward, forward_pif_ch);
                }
            }
        }
        // for (const auto& f : forward) {
        //   std::cout << "(" << f.size() << ", " << f.front().size() << "), ";
        // }
        // std::cout << '\n';
        // for (const auto& f : backward) {
        //   std::cout << "(" << f.size() << ", " << f.front().size() << "), ";
        // }
        // std::cout << '\n';
        // END: caf_scored = utils.CafScored(cifhr.accumulated).fill(fields, self.caf_metas)
        std::sort(seeds.begin(), seeds.end(), std::greater{});

        // occupacy map.
        // std::cout << C << ' ' << H_hr << ' ' << W_hr << '\n';
        Occupancy occupied(C, H_hr, W_hr);
        std::vector<Annotation> annotations;
        for (const auto& [v, f, x, y, s] : seeds) {
            if (occupied.fuzz_get(f, y, x)) {
                continue;
            }

            Annotation ann(f, x, y, v);
            ann.jointScales[f] = s;
            grow(ann, forward, backward);
            annotations.push_back(ann);

            for (int i = 0; i < N_PIFPAF_KEYPOINTS; ++i) {
                const auto ax = ann.keypoints[i * 3];
                const auto ay = ann.keypoints[i * 3 + 1];
                const auto av = ann.keypoints[i * 3 + 2];
                if (av == 0) {
                    continue;
                }

                const auto width = ann.jointScales[i];
                scalarSquareAddSingle(occupied, i, H_hr, W_hr, ax, ay, width, Occupancy::reduction, Occupancy::min_scale_reduced); // width is sigma...
            }
        }

        // This returns two lists that each contain 19 tensors of shape (7, ?)
        // where the second dimension can vary in size (depends on thresholds).
        // const auto pt = scorePafTarget(paf, pifhr);
        // const auto pafForward = std::get<0>(pt);
        // const auto pafBackward = std::get<1>(pt);

        /*
  // For debugging
  printf("pafForward:\n");
  for (auto& i : pafForward) {
    for (auto j : i) { printf("%f, ", j); } printf("\n");
  }
  printf("\npafBackward:\n");
  for (auto i : pafBackward) {
    for (auto& j : i) { printf("%f, ", j); } printf("\n");
  }
  */

        // auto annotations = decodeAnnotations(seeds, pifhr, pifhrScales, pifhrCore, pafForward, pafBackward);

        // Scale to input size
        //  for (auto& ann : annotations) {
        //    for (int k = 0; k < N_PIFPAF_KEYPOINTS; ++k) {
        //      ann.keypoints[k*3    ] *= STRIDE;
        //      ann.keypoints[k*3 + 1] *= STRIDE;
        //        std::cout << "--> Scaled: " <<ann.keypoints[k*3    ] << '\t' << ann.keypoints[k*3+1] << '\n';
        //      ann.jointScales[k]     *= STRIDE;
        //    }
        //  }

        // Non-maximum suppression
        if (!annotations.empty()) {
            annotations = softNMS(annotations);
        }

        // // Threshold
        std::vector<Annotation> thresholded;
        for (auto& ann : annotations) {
            for (int k = 0; k < N_PIFPAF_KEYPOINTS; ++k) {
                if (ann.keypoints[k * 3 + 2] < keypointThreshold) {
                    ann.keypoints[k * 3 + 2] = 0.0f;
                }
            }
            if (ann.score() >= instanceThreshold) {
                thresholded.push_back(ann);
            }
        }

        std::sort(thresholded.begin(), thresholded.end(), [](const Annotation& a, const Annotation& b) {
            return a.score() > b.score();
        });

        // // Convert to normalized coordinates
        //  for (auto& ann : thresholded) {
        //    for (int k = 0; k < N_PIFPAF_KEYPOINTS; ++k) {
        //      ann.keypoints[k*3    ] /= inputWidth;
        //      ann.keypoints[k*3 + 1] /= inputHeight;
        //    }
        //  }

        /*
  // For debugging
  for (auto ann : thresholded) {
    printf("Keypoints:\n");
    for (auto k : ann.keypoints) {
      printf("%f, ", k);
    }
    printf("\nJoint scales:\n");
    for (auto k : ann.jointScales) {
      printf("%f, ", k);
    }
    printf("\n");
  }
  */

        ai_app::Object_detection::Result result;
        result.success = true;
        for (auto& ann : thresholded) {
            ai_app::Landmarks landmarks;
            landmarks.type = "body_pose_pifpaf";

            int minx = std::numeric_limits<int>::max(),
                miny = std::numeric_limits<int>::max(),
                maxx_ = std::numeric_limits<int>::min(),
                maxy_ = std::numeric_limits<int>::min();

            for (int k = 0; k < N_PIFPAF_KEYPOINTS; ++k) {
                const int x = ann.keypoints[k * 3];
                const int y = ann.keypoints[k * 3 + 1];
                const auto v = ann.keypoints[k * 3 + 2];

                if (v > 0.0f) {
                    if (x < minx) {
                        minx = x;
                    }
                    if (x > maxx_) {
                        maxx_ = x;
                    }
                    if (y < miny) {
                        miny = y;
                    }
                    if (y > maxy_) {
                        maxy_ = y;
                    }
                }

                ai_app::Landmark landmark{};
                landmark.confidence = v;
                landmark.position.x = x;
                landmark.position.y = y;
                landmarks.points.push_back(landmark);
            }

            ai_app::Object_detection::Result::Item item;
            item.confidence = ann.score();
            item.class_index = 1;
            item.bounding_box.origin.x = minx;
            item.bounding_box.origin.y = miny;
            item.bounding_box.size.x = maxx_ - minx;
            item.bounding_box.size.y = maxy_ - miny;
            item.landmarks = landmarks;

            result.items.push_back(item);
        }
        return result;
    }

}
}
