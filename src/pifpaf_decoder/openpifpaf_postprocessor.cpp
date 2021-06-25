#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <cstring>
#include "openpifpaf_postprocessor.hpp"
#include "math_helpers.hpp"

namespace lpdnn::aiapp_impl {

const int OpenPifPafPostprocessor::bones[19][2] = {
  {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, { 6, 12}, { 7, 13},
  { 6,  7}, { 6,  8}, { 7,  9}, { 8, 10}, { 9, 11}, { 2,  3}, { 1,  2},
  { 1,  3}, { 2,  4}, { 3,  5}, { 4,  6}, { 5,  7},
};

constexpr int C = 17;
constexpr float stride = 8.0f;
constexpr float seedThreshold = 0.2f;
constexpr float keypointThreshold = 0.001f;
constexpr float instanceThreshold = 0.2f;

/*
  Creates a (2, h, w) tensor where the first part is:
      0, 1, 2, 3, ..., w-1,
      0, 1, 2, 3, ..., w-1,
      0, 1, 2, 3, ..., w-1,
      ...
  and the second part is:
      0, 0, 0, 0, ..., 0,
      1, 1, 1, 1, ..., 1,
      2, 2, 2, 2, ..., 2,
      ...
  Used for normaling the PIFs and PAFs.
*/
static std::vector<float> makeIndexField(int h, int w) {
  std::vector<float> indexField(2 * h * w);
  float* ptr = indexField.data();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      ptr[ y     *w + x] = (float)x;
      ptr[(y + h)*w + x] = (float)y;
    }
  }
  return indexField;
}

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

static void scalarSquareAddGauss(float* field,
                                 int fieldH,
                                 int fieldW,
                                 const std::vector<float>& x,
                                 const std::vector<float>& y,
                                 const std::vector<float>& sigma_,
                                 const std::vector<float>& v,
                                 float truncate = 2.0f)
{
    // sigma_np = np.maximum(1.0, sigma_np)
    // width_np = np.maximum(1.0, truncate * sigma_np)
    auto sigma = sigma_;
    std::vector<float> width(sigma.size());
    for (size_t i = 0; i < sigma.size(); ++i) {
        sigma[i] = std::max(1.0f, sigma[i]);
        width[i] = std::max(1.0f, truncate * sigma[i]);
    }

    // NOTE: The minx, miny, maxx, maxxy code is the same as in scalarSquareAddConstant().
    // Could probably extract that and do it just once.

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
    //         deltax = xx - x[i]
    //         for yy in range(miny[i], maxy[i]):
    //             deltay = yy - y[i]
    //             vv = v[i] * np.exp(-0.5 * (deltax**2 + deltay**2) / sigma[i]**2)
    //             field[yy, xx] += vv
    for (size_t i = 0; i < minx.size(); ++i) {
        for (int xx = minx[i]; xx < maxx[i]; ++xx) {
            float deltax = (float)xx - x[i];
            for (int yy = miny[i]; yy < maxy[i]; ++yy) {
                float deltay = (float)yy - y[i];
                float vv = v[i] * std::exp(-0.5f * (deltax*deltax + deltay*deltay) / (sigma[i]*sigma[i]));
                field[yy * fieldW + xx] += vv;
            }
        }
    }

    /*
    // For debugging
    for (int y = 0; y < fieldH; ++y) {
      for (int x = 0; x <fieldW; ++x) {
        printf("%f, ", field[y*fieldW + x]);
      }
      printf("\n");
    }
    */
}

static void scalarSquareAddSingle(float* field,
                                  int fieldH,
                                  int fieldW,
                                  float x,
                                  float y,
                                  float width,
                                  float value)
{
    // minx = max(0, int(round(x - width)))
    // miny = max(0, int(round(y - width)))
    auto minx = std::max(0, (int)std::round(x - width));
    auto miny = std::max(0, (int)std::round(y - width));

    // maxx = max(minx + 1, min(field.shape[1], int(round(x + width)) + 1))
    // maxy = max(miny + 1, min(field.shape[0], int(round(y + width)) + 1))
    auto maxx = std::max(minx + 1, std::min(fieldW, (int)std::round(x + width) + 1));
    auto maxy = std::max(miny + 1, std::min(fieldH, (int)std::round(y + width) + 1));

    if (minx >= fieldW) { return; }
    if (miny >= fieldH) { return; }

    // field[miny:maxy, minx:maxx] += value
    for (auto yy = miny; yy < maxy; ++yy) {
        for (auto xx = minx; xx < maxx; ++xx) {
            field[yy * fieldW + xx] += value;
        }
    }
}

/**
  Combines the different PAF outputs into one big (19, 2, 4, h, w) tensor.

  The input tensors have the shape (19, h, w) except for j1/j2Fields, which
  are (38, h, w).
*/
void OpenPifPafPostprocessor::normalizePAF(const float* intensityFields,
                                           const float* j1Fields,
                                           const float* j2Fields,
                                           const float* j1FieldsLogb,
                                           const float* j2FieldsLogb)
{
    float* pafPtr = paf.data();

    // Strides for the first dimension of the input tensors:
    const size_t if_stride_0   = H * W;
    const size_t j1f_stride_0  = H * W;
    const size_t j1bf_stride_0 = H * W;
    const size_t j2f_stride_0  = H * W;
    const size_t j2bf_stride_0 = H * W;

    for (int i = 0; i < 19; ++i) {
        // Copy the next h*w values from intensityFields.
        size_t ifOffset = i * if_stride_0;
        size_t outOffset = i * paf_stride_0;
        memcpy(pafPtr + outOffset, intensityFields + ifOffset, H * W * sizeof(float));

        // Copy the next 2 h*w values from j1Fields.
        size_t j1fOffset = (i * 2) * j1f_stride_0;
        outOffset += paf_stride_2;
        memcpy(pafPtr + outOffset, j1Fields + j1fOffset, 2 * H * W * sizeof(float));

        // Also add the index field to the values from j1Fields.
        vadd(indexField.data(), j1Fields + j1fOffset, pafPtr + outOffset, 2 * H * W);

        // Copy the next h*w values from j1FieldsLogb and exponentiate.
        size_t j1bfOffset = i * j1bf_stride_0;
        outOffset += paf_stride_2 * 2;
        memcpy(pafPtr + outOffset, j1FieldsLogb + j1bfOffset, H * W * sizeof(float));
        vexp(pafPtr + outOffset, H * W);

        // Copy the same h*w values from intensityFields again.
        outOffset = i * paf_stride_0 + paf_stride_1;
        memcpy(pafPtr + outOffset, intensityFields + ifOffset, H * W * sizeof(float));

        // Copy the next 2 h*w values from j2Fields.
        size_t j2fOffset = (i * 2) * j2f_stride_0;
        outOffset += paf_stride_2;
        memcpy(pafPtr + outOffset, j2Fields + j2fOffset, 2 * H * W * sizeof(float));

        // Also add the index field to the values from j2Fields.
        vadd(indexField.data(), j2Fields + j2fOffset, pafPtr + outOffset, 2 * H * W);

        // Copy the next h*w values from j2FieldsLogb and exponentiate.
        size_t j2bfOffset = i * j2bf_stride_0;
        outOffset += paf_stride_2 * 2;
        memcpy(pafPtr + outOffset, j2FieldsLogb + j2bfOffset, H * W * sizeof(float));
        vexp(pafPtr + outOffset, H * W);
    }

    // NOTE: We could do the exponentiation for j1/j2FieldsLogb in the Core ML
    // model already.

    /*
    // For debugging
    for (int y = 0; y < H; ++y) {
      printf("%d: ", y);
      for (int x = 0; x < W; ++x) {
        printf("%f, ", paf[9*paf_stride_0 + 2*paf_stride_1 + 7*paf_stride_2 + y*W + x]);
      }
      printf("\n");
    }
    */
}

/**
  Combines the different PIF outputs into one big (17, 4, h, w) tensor.

  The input tensors have the shape (17, h, w) except for jointFields, which
  is (34, h, w).
*/
void OpenPifPafPostprocessor::normalizePIF(const float* jointIntensityFields,
                                           const float* jointFields,
                                           const float* scaleFields)
{
    float* pifPtr = pif.data();

    // Strides for the first dimension of the input tensors:
    const size_t iif_stride_0 = H * W;
    const size_t jf_stride_0  = H * W;
    const size_t sf_stride_0  = H * W;

    // The PyTorch code concatenates the following tensors:
    //   (17, 1, h, w)
    //   (17, 2, h, w)
    //   (17, 1, h, w)
    // along the 2nd axis into one tensor of shape (17, 4, h, w). But the
    // tensors from Core ML have the following shapes:
    //   (17, h, w)
    //   (34, h, w)
    //   (17, h, w)
    // Fortunately, (17, 2, ...) has the same memory layout as (34, ...),
    // so we can simply do a bunch of memcpy's.

    for (int i = 0; i < 17; ++i) {
        // Copy the next h*w values from jointIntensityFields.
        size_t jifOffset = i * iif_stride_0;
        size_t outOffset = i * pif_stride_0;
        memcpy(pifPtr + outOffset, jointIntensityFields + jifOffset, H * W * sizeof(float));

        // Copy the next 2 h*w values from jointFields.
        size_t jfOffset = (i * 2) * jf_stride_0;
        outOffset += pif_stride_1;
        memcpy(pifPtr + outOffset, jointFields + jfOffset, 2 * H * W * sizeof(float));

        // Also add the index field to the values from jointFields.
        vadd(indexField.data(), jointFields + jfOffset, pifPtr + outOffset, 2 * H * W);

        // Copy the next h*w values from scaleFields.
        size_t sfOffset = i * sf_stride_0;
        outOffset += pif_stride_1 * 2;
        memcpy(pifPtr + outOffset, scaleFields + sfOffset, H * W * sizeof(float));
    }
}

OpenPifPafPostprocessor::Target_intensity
OpenPifPafPostprocessor::targetIntensities(const std::vector<float>& pif,
                                           float v_th, bool coreOnly)
{
    const float pif_nn = 16.0f;

    const size_t targets_stride_0 = H_hr * W_hr;
    const size_t scales_stride_0  = H_hr * W_hr;
    const size_t ns_stride_0      = H_hr * W_hr;

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
        const size_t sOffset = yOffset + pif_stride_1;
        for (int j = 0; j < H*W; ++j) {
            float p = pif[pifOffset + j];
            if (p > v_th) {
                v.push_back(p);
                x.push_back(pif[xOffset + j] * stride);
                y.push_back(pif[yOffset + j] * stride);
                s.push_back(pif[sOffset + j] * stride);
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

        // v / pif_nn
        std::vector<float> v_over_pif_nn(v.size());
        vsmul(v.data(), 1.0f / pif_nn, v_over_pif_nn.data(), v.size());

        // The original code computed the "core only" version in a separate step
        // but that duplicates a bunch of work, so we do it at the same time.
        const auto tco = targetsCoreOnly.data() + i * targets_stride_0;
        scalarSquareAddGauss(tco, H_hr, W_hr, x, y, s, v_over_pif_nn, 0.5);

        // s * v
        std::vector<float> s_times_v(v.size());
        vmul(s.data(), v.data(), s_times_v.data(), v.size());

        const auto t = targets.data() + i * targets_stride_0;
        const auto scale = scales.data() + i * scales_stride_0;
        const auto n = ns.data() + i * ns_stride_0;
        scalarSquareAddGauss(t, H_hr, W_hr, x, y, s, v_over_pif_nn);
        scalarSquareAddConstant(scale, H_hr, W_hr, x, y, s, s_times_v);
        scalarSquareAddConstant(n, H_hr, W_hr, x, y, s, v);
    }

    // m = ns > 0
    // scales[m] = scales[m] / ns[m]
    for (size_t i = 0; i < scales.size(); ++i) {
        const auto d = ns[i];
        if (d > 0) { scales[i] /= d; }
    }

    return Target_intensity{ targets, scales, targetsCoreOnly };
}

OpenPifPafPostprocessor::Paf_target
OpenPifPafPostprocessor::scorePafTarget(const std::vector<float>& pafvec,
                                        const std::vector<float>& pifhr,
                                        float pifhr_floor,
                                        float score_th) const
{
    std::vector<std::vector<float>> scored_forward;
    std::vector<std::vector<float>> scored_backward;

    for (int c = 0; c < 19; ++c) {
        // The PAF has shape (19, 2, 4, h, w). We're looking at one (2, 4, h, w)
        // slice at a time in this loop.
        const size_t pafOffset = c * paf_stride_0;

        // scores = np.min(fourds[:, 0], axis=0)
        // mask = scores > score_th
        // scores = scores[mask]
        std::vector<float> scores;
        std::vector<int> mask;
        for (int i = 0; i < H * W; ++i) {
            auto a = pafvec[pafOffset + i];
            auto b = pafvec[pafOffset + paf_stride_1 + i];
            auto score = std::min(a, b);
            if (score > score_th) {
                scores.push_back(score);
                mask.push_back(i);
            }
        }

        // fourds = fourds[:, :, mask]
        const size_t scores_size = scores.size();
        std::vector<float> masked(2 * 4 * scores_size);
        for (size_t i = 0; i < mask.size(); ++i) {
            const auto m = mask[i];
            masked[i                ] = pafvec[pafOffset                                 + m];
            masked[i + scores_size  ] = pafvec[pafOffset                + paf_stride_2   + m];
            masked[i + scores_size*2] = pafvec[pafOffset                + paf_stride_2*2 + m];
            masked[i + scores_size*3] = pafvec[pafOffset                + paf_stride_2*3 + m];
            masked[i + scores_size*4] = pafvec[pafOffset + paf_stride_1                  + m];
            masked[i + scores_size*5] = pafvec[pafOffset + paf_stride_1 + paf_stride_2   + m];
            masked[i + scores_size*6] = pafvec[pafOffset + paf_stride_1 + paf_stride_2*2 + m];
            masked[i + scores_size*7] = pafvec[pafOffset + paf_stride_1 + paf_stride_2*3 + m];
        }

        std::vector<float> scores_b(scores_size);
        if (pifhr_floor < 1.0f) {
            // ij_b = np.round(fourds[0, 1:3] * self.stride).astype(np.int)
            // ij_b[0] = np.clip(ij_b[0], 0, self._pifhr.shape[2] - 1)
            // ij_b[1] = np.clip(ij_b[1], 0, self._pifhr.shape[1] - 1)
            std::vector<int> ij_b(2 * scores_size);
            for (size_t i = 0; i < scores_size*2; ++i) {
                const int v = (int)std::round(masked[scores_size + i] * stride);
                ij_b[i] = std::min(std::max(0, v), i < scores_size ? W_hr - 1 : H_hr - 1);
            }

            // pifhr_b = self._pifhr[j1i, ij_b[1], ij_b[0]]
            // scores_b = scores * (pifhr_floor + (1.0 - pifhr_floor) * pifhr_b)
            const auto j1i = bones[c][0] - 1;
            for (size_t i = 0; i < scores_b.size(); ++i) {
                const auto pifhr_b = pifhr[j1i * pifhr_stride_0 + ij_b[scores_size + i] * pifhr_stride_1 + ij_b[i]];
                scores_b[i] = scores[i] * (pifhr_floor + (1.0f - pifhr_floor) * pifhr_b);
            }
        } else {
            scores_b = scores;
        }

        // mask_b = scores_b > score_th
        std::vector<int> mask_b;
        for (int i = 0; i < (int)scores_b.size(); ++i) {
            if (scores_b[i] > score_th) { mask_b.push_back(i); }
        }

        const size_t mask_b_size = mask_b.size();
        std::vector<float> result_b(7 * mask_b_size);
        for (size_t i = 0; i < mask_b_size; ++i) {
            const auto m = mask_b[i];
            result_b[i                ] = scores_b[m];
            result_b[i + mask_b_size  ] = masked[scores_size*5 + m];
            result_b[i + mask_b_size*2] = masked[scores_size*6 + m];
            result_b[i + mask_b_size*3] = masked[scores_size*7 + m];
            result_b[i + mask_b_size*4] = masked[scores_size   + m];
            result_b[i + mask_b_size*5] = masked[scores_size*2 + m];
            result_b[i + mask_b_size*6] = masked[scores_size*3 + m];
        }
        scored_backward.push_back(result_b);

        std::vector<float> scores_f(scores_size);
        if (pifhr_floor < 1.0f) {
            // ij_f = np.round(fourds[1, 1:3] * self.stride).astype(np.int)
            // ij_f[0] = np.clip(ij_f[0], 0, self._pifhr.shape[2] - 1)
            // ij_f[1] = np.clip(ij_f[1], 0, self._pifhr.shape[1] - 1)
            std::vector<int> ij_f(2 * scores_size);
            for (size_t i = 0; i < scores_size*2; ++i) {
                const int v = (int)std::round(masked[scores_size*5 + i] * stride);
                ij_f[i] = std::min(std::max(0, v), i < scores_size ? W_hr - 1 : H_hr - 1);
            }

            // pifhr_f = self._pifhr[j2i, ij_f[1], ij_f[0]]
            // scores_f = scores * (pifhr_floor + (1.0 - pifhr_floor) * pifhr_f)
            const auto j2i = bones[c][1] - 1;
            for (size_t i = 0; i < scores_f.size(); ++i) {
                const auto pifhr_f = pifhr[j2i * pifhr_stride_0 + ij_f[scores_size + i] * pifhr_stride_1 + ij_f[i]];
                scores_f[i] = scores[i] * (pifhr_floor + (1.0f - pifhr_floor) * pifhr_f);
            }
        } else {
            scores_f = scores;
        }

        // mask_f = scores_f > score_th
        std::vector<int> mask_f;
        for (int i = 0; i < (int)scores_b.size(); ++i) {
            if (scores_f[i] > score_th) { mask_f.push_back(i); }
        }

        // scored_forward.append(np.concatenate((
        //     np.expand_dims(scores_f[mask_f], 0),
        //     fourds[0, 1:4][:, mask_f],
        //     fourds[1, 1:4][:, mask_f],
        // )))
        const size_t mask_f_size = mask_f.size();
        std::vector<float> result_f(7 * mask_f_size);
        for (size_t i = 0; i < mask_f_size; ++i) {
            const auto m = mask_f[i];
            result_f[i                ] = scores_f[m];
            result_f[i + mask_f_size  ] = masked[scores_size   + m];
            result_f[i + mask_f_size*2] = masked[scores_size*2 + m];
            result_f[i + mask_f_size*3] = masked[scores_size*3 + m];
            result_f[i + mask_f_size*4] = masked[scores_size*5 + m];
            result_f[i + mask_f_size*5] = masked[scores_size*6 + m];
            result_f[i + mask_f_size*6] = masked[scores_size*7 + m];
        }
        scored_forward.push_back(result_f);

        /*
        // For debugging
        printf("iteration: %d\n", c);
        printf("scores:\n"); for (auto n : scores) printf("%f, ", n); printf("\n");
        printf("mask:\n"); for (auto n : mask) printf("%d, ", n); printf("\n");
        printf("masked:\n"); for (auto n : masked) printf("%f, ", n); printf("\n");
        printf("scores_b:\n"); for (auto n : scores_b) printf("%f, ", n); printf("\n");
        printf("scores_f:\n"); for (auto n : scores_f) printf("%f, ", n); printf("\n");
        */
    }
    return Paf_target{ scored_forward, scored_backward };
}

std::vector<OpenPifPafPostprocessor::Pifhr_seed>
OpenPifPafPostprocessor::pifhrSeeds(const std::vector<float>& pifhrScales,
                                    const std::vector<float>& pifhrCore)
{
    std::vector<Pifhr_seed> seeds;

    for (int field_i = 0; field_i < 17; ++field_i) {
        const size_t pifhrScalesOffset = field_i * pifhr_stride_0;
        const size_t pifhrCoreOffset = field_i * pifhr_stride_0;

        // candidates = np.concatenate((index_fields, np.expand_dims(f, 0)), 0)
        // mask = f > self.seed_threshold
        std::vector<int> mask;
        for (int i = 0; i < H_hr * W_hr; ++i) {
            const auto value = pifhrCore[pifhrCoreOffset + i];
            if (value > seedThreshold) { mask.push_back(i); }
        }

        // candidates = np.moveaxis(candidates[:, mask], 0, -1)
        // This is a (count, 3) tensor where count is #elements over threshold.
        std::vector<float> masked(mask.size() * 3);
        for (size_t i = 0; i < mask.size(); ++i) {
            const auto m = mask[i];
            masked[i*3    ] = indexField_hr[m];
            masked[i*3 + 1] = indexField_hr[m + H_hr*W_hr];
            masked[i*3 + 2] = pifhrCore[pifhrCoreOffset + m];
        }

        // occupied = np.zeros(s.shape)
        std::vector<float> occupied(H_hr * W_hr, 0.0f);

        std::vector<int> sorted(mask.size());
        std::iota(sorted.begin(), sorted.end(), 0);
        std::sort(sorted.begin(), sorted.end(), [masked] (int const& a, int const& b) {
            return masked[a*3 + 2] > masked[b*3 + 2];
        });

        // for c in sorted(candidates, key=lambda c: c[2], reverse=True):
        for (auto c : sorted) {
            const auto c_0 = masked[c*3];
            const auto c_1 = masked[c*3 + 1];
            const auto c_2 = masked[c*3 + 2];

            // i, j = int(c[0]), int(c[1])
            const auto i = (int)c_0;
            const auto j = (int)c_1;
            if (occupied[j*W_hr + i] > 0) { continue; }

            // width = max(4, s[j, i])
            const auto s = pifhrScales[pifhrScalesOffset + j * pifhr_stride_1 + i];
            const auto width = std::max(4.0f, s);

            // scalar_square_add_single(occupied, c[0], c[1], width / 2.0, 1.0)
            scalarSquareAddSingle(occupied.data(), H_hr, W_hr, c_0, c_1, width / 2.0f, 1.0f);

            // seeds.append((c[2], field_i, c[0] / self.stride, c[1] / self.stride))
            seeds.emplace_back( c_2, field_i, c_0 / stride, c_1 / stride );
        }
    }

    // seeds = list(sorted(seeds, reverse=True))
    std::sort(seeds.begin(), seeds.end(), [] (const Pifhr_seed& a, const Pifhr_seed& b) {
        const auto ca = std::get<0>(a);
        const auto cb = std::get<0>(b);
        return ca > cb;
    });

    // if len(seeds) > 500:
    //     if seeds[500][0] > 0.1:
    //         seeds = [s for s in seeds if s[0] > 0.1]
    //     else:
    //         seeds = seeds[:500]
    if (seeds.size() > 500) {
        seeds.resize(500);
    }
    return seeds;
}

std::vector<float>
OpenPifPafPostprocessor::pafCenter(const std::vector<float>& paf_field,
                                   float x, float y, float sigma)
{
    std::vector<int> mask;
    const int paf_stride = (int)paf_field.size() / 7;
    for (int i = 0; i < paf_stride; ++i) {
        const bool take = (paf_field[  paf_stride + i] > x - sigma * paf_field[3*paf_stride + i]) &&
                          (paf_field[  paf_stride + i] < x + sigma * paf_field[3*paf_stride + i]) &&
                          (paf_field[2*paf_stride + i] > y - sigma * paf_field[3*paf_stride + i]) &&
                          (paf_field[2*paf_stride + i] < y + sigma * paf_field[3*paf_stride + i]);
        if (take) { mask.push_back(i); }
    }
    if (mask.empty()) { return {}; }

    const int mask_size = (int)mask.size();
    const int out_stride = mask_size;
    std::vector<float> result(7 * mask_size, 0.0f);
    for (int j = 0; j < 7; ++j) {
        for (int i = 0; i < mask_size; ++i) {
            const int m = mask[i];
            result[j*out_stride + i] = paf_field[j*paf_stride + m];
        }
    }
    return result;
}

OpenPifPafPostprocessor::Connection
OpenPifPafPostprocessor::growConnection(float x, float y,
                                        const std::vector<float>& paf_field_)
{
    // # source value
    // paf_field = paf_center(paf_field, xy[0], xy[1], sigma=2.0)
    // if paf_field.shape[1] == 0:
    //     return 0, 0, 0
    const auto paf_field = pafCenter(paf_field_, x, y, 2.0f);
    if (paf_field.empty()) { return Connection{ 0, 0, 0}; }

    // # source distance
    // d = np.linalg.norm(np.expand_dims(xy, 1) - paf_field[1:3], axis=0)
    // b_source = paf_field[3] * 3.0
    // # combined value and source distance
    // v = paf_field[0]
    // scores = np.exp(-1.0 * d / b_source) * v  # two-tailed cumulative Laplace
    const int paf_stride = (int)paf_field.size() / 7;
    std::vector<float> scores(paf_stride);
    for (int i = 0; i < paf_stride; ++i) {
        const auto a = x - paf_field[paf_stride   + i];
        const auto b = y - paf_field[paf_stride*2 + i];
        const auto d = std::sqrt(a*a + b*b);
        const auto b_source = paf_field[paf_stride*3 + i] * 3.0f;
        const auto v = paf_field[i];
        scores[i] = std::exp(-d / b_source) * v;
    }

    // return self._target_with_maxscore(paf_field[4:7], scores)
    int max_i;
    const float score = vargmax(scores.data(), scores.size(), &max_i);
    return Connection{ paf_field[paf_stride*4 + max_i], paf_field[paf_stride*5 + max_i], score };
}

std::vector<OpenPifPafPostprocessor::frontier_t> OpenPifPafPostprocessor::frontier(Annotation& ann) {
    std::vector<frontier_t> f;

    for (int connection_i = 0; connection_i < numBones; ++connection_i) {
        const auto bone = bones[connection_i];
        const auto j1i = bone[0] - 1;
        const auto j2i = bone[1] - 1;
        if (ann.keypoints[j1i*3 + 2] > 0.0f && ann.keypoints[j2i*3 + 2] == 0.0f) {
            f.emplace_back( ann.keypoints[j1i*3 + 2], connection_i, true, j1i, j2i );
        }
    }

    for (int connection_i = 0; connection_i < numBones; ++connection_i) {
        const auto bone = bones[connection_i];
        const auto j1i = bone[0] - 1;
        const auto j2i = bone[1] - 1;
        if (ann.keypoints[j2i*3 + 2] > 0.0f && ann.keypoints[j1i*3 + 2] == 0.0f) {
            f.emplace_back( ann.keypoints[j2i*3 + 2], connection_i, false, j1i, j2i );
        }
    }

    std::sort(f.begin(), f.end(), [] (const frontier_t& a, const frontier_t& b) {
        const auto ca = std::get<0>(a);
        const auto cb = std::get<0>(b);
        return ca > cb;
    });

    return f;
}

OpenPifPafPostprocessor::frontier_t OpenPifPafPostprocessor::frontierIter(Annotation& ann) {
    while (frontierActive) {
        // unblocked_frontier = [f for f in self.frontier()
        //                       if (f[1], f[2]) not in block_frontier]
        std::vector<frontier_t> unblockedFrontier;
        for (auto f : frontier(ann)) {
            const auto connection_id = std::get<1>(f);
            const auto forward = std::get<2>(f);
            if (blockFrontier.find(std::tuple<int, bool>{ connection_id, forward }) == blockFrontier.end()) {
                unblockedFrontier.push_back(f);
            }
        }

        /*
        // For debugging
        printf("unblockedFrontier ");
        for (auto n : unblockedFrontier) {
          printf("(%f, %d, %s, %d, %d), ", std::get<0>(n), std::get<1>(n),
                                           std::get<2>(n) ? "true" : "false",
                                           std::get<3>(n), std::get<4>(n));
        }
        printf("\n");
        */

        // if not unblocked_frontier:
        //     break
        if (unblockedFrontier.empty()) {
            frontierActive = false;
            break;
        }

        // first = unblocked_frontier[0]
        // yield first
        // block_frontier.add((first[1], first[2]))
        const auto first = unblockedFrontier[0];
        const auto connection_id = std::get<1>(first);
        const auto forward = std::get<2>(first);
        blockFrontier.insert(std::tuple<int, bool>{ connection_id, forward });
        return first;
    }
    return {};
}

void OpenPifPafPostprocessor::grow(Annotation& ann,
                                   const std::vector<std::vector<float>>& pafForward,
                                   const std::vector<std::vector<float>>& pafBackward,
                                   float th)
{
    frontierActive = true;
    blockFrontier.clear();

    while (true) {
        const auto f = frontierIter(ann);
        if (!frontierActive) { return; }

        const auto i = std::get<1>(f);
        const auto forward = std::get<2>(f);
        const auto j1i = std::get<3>(f);
        const auto j2i = std::get<4>(f);

        // For debugging
        //printf("grow: %d %s %d %d\n", i, forward ? "true" : "false", j1i, j2i);

        float x, y, v;
        std::vector<float> directed_paf_field;
        std::vector<float> directed_paf_field_reverse;
        if (forward) {
            x = ann.keypoints[j1i*3    ];
            y = ann.keypoints[j1i*3 + 1];
            v = ann.keypoints[j1i*3 + 2];
            directed_paf_field = pafForward[i];
            directed_paf_field_reverse = pafBackward[i];
        } else {
            x = ann.keypoints[j2i*3    ];
            y = ann.keypoints[j2i*3 + 1];
            v = ann.keypoints[j2i*3 + 2];
            directed_paf_field = pafBackward[i];
            directed_paf_field_reverse = pafForward[i];
        }

        const auto t = growConnection(x, y, directed_paf_field);
        const auto new_x = std::get<0>(t);
        const auto new_y = std::get<1>(t);
        auto new_v = std::get<2>(t);

        if (new_v < th) { continue; }

        // reverse match
        if (th >= 0.1) {
            const auto t1 = growConnection(new_x, new_y, directed_paf_field_reverse);
            const auto reverse_x = std::get<0>(t1);
            const auto reverse_y = std::get<1>(t1);
            const auto reverse_v = std::get<2>(t1);
            if (reverse_v < th) { continue; }
            if (std::abs(x - reverse_x) + std::abs(y - reverse_y) > 1.0f) { continue; }
        }

        new_v = std::sqrt(new_v * v);  // geometric mean

        if (forward) {
            if (new_v > ann.keypoints[j2i*3 + 2]) {
                ann.keypoints[j2i*3    ] = new_x;
                ann.keypoints[j2i*3 + 1] = new_y;
                ann.keypoints[j2i*3 + 2] = new_v;
            }
        } else {
            if (new_v > ann.keypoints[j1i*3 + 2]) {
                ann.keypoints[j1i*3    ] = new_x;
                ann.keypoints[j1i*3 + 1] = new_y;
                ann.keypoints[j1i*3 + 2] = new_v;
            }
        }
    }
}

void OpenPifPafPostprocessor::fillJointScales(Annotation& ann,
                                              const std::vector<float>& scales,
                                              int fieldH,
                                              int fieldW,
                                              float hr_scale)
{
    for (int k = 0; k < numKeypoints; ++k) {
        const auto x = ann.keypoints[k*3];
        const auto y = ann.keypoints[k*3 + 1];
        const auto v = ann.keypoints[k*3 + 2];
        if (v == 0) { continue; }

        // i = max(0, min(scale_field.shape[1] - 1, int(round(xyv[0] * hr_scale))))
        // j = max(0, min(scale_field.shape[0] - 1, int(round(xyv[1] * hr_scale))))
        const auto i = std::max(0, std::min(fieldW - 1, (int)std::round(x * hr_scale)));
        const auto j = std::max(0, std::min(fieldH - 1, (int)std::round(y * hr_scale)));

        // self.joint_scales[xyv_i] = scale_field[j, i] / hr_scale
        ann.jointScales[k] = scales[k*pifhr_stride_0 + j*pifhr_stride_1 + i] / hr_scale;
    }
}

std::vector<OpenPifPafPostprocessor::Annotation>
OpenPifPafPostprocessor::decodeAnnotations(const std::vector<float>& pifhr,
                                           const std::vector<float>& pifhrScales,
                                           const std::vector<float>& pifhrCore,
                                           const std::vector<std::vector<float>>& pafForward,
                                           const std::vector<std::vector<float>>& pafBackward)
{
    const auto seeds = pifhrSeeds(pifhrScales, pifhrCore);

    // This is a (17, H_hr, W_hr) tensor.
    std::vector<float> occupied(17 * H_hr * W_hr, 0.0f);

    std::vector<Annotation> annotations;
    for (auto& seed : seeds) {
        const auto v = std::get<0>(seed);
        const auto f = std::get<1>(seed);
        const auto x = std::get<2>(seed);
        const auto y = std::get<3>(seed);

        const auto i = std::min(std::max(0, (int)std::round(x * stride)), W_hr - 1);
        const auto j = std::min(std::max(0, (int)std::round(y * stride)), H_hr - 1);
        if (occupied[f*H_hr*W_hr + j*W_hr + i] > 0.0f) { continue; }

        Annotation ann(f, x, y, v);
        grow(ann, pafForward, pafBackward);
        fillJointScales(ann, pifhrScales, H_hr, W_hr, stride);
        annotations.push_back(ann);

        for (int i = 0; i < numKeypoints; ++i) {
            const auto x = ann.keypoints[i*3];
            const auto y = ann.keypoints[i*3 + 1];
            const auto v = ann.keypoints[i*3 + 2];
            if (v == 0) { continue; }

            const auto width = ann.jointScales[i] * stride;
            scalarSquareAddSingle(occupied.data() + i*H_hr*W_hr, H_hr, W_hr,
                                  x * stride, y * stride, width / 2.0f, 1.0f);
        }
    }
    return annotations;
}

std::vector<OpenPifPafPostprocessor::Annotation> OpenPifPafPostprocessor::softNMS(std::vector<Annotation>& annotations) {
    float maxx = 0.0f;
    float maxy = 0.0f;
    for (auto& ann : annotations) {
        for (int k = 0; k < numKeypoints; ++k) {
            auto x = ann.keypoints[k*3];
            auto y = ann.keypoints[k*3 + 1];
            if (x > maxx) { maxx = x; }
            if (y > maxy) { maxy = y; }
        }
    }

    const auto h = (int)(maxy + 1);
    const auto w = (int)(maxx + 1);
    std::vector<float> occupied(17 * h * w, 0.0f);

    std::vector<int> sorted(annotations.size());
    std::iota(sorted.begin(), sorted.end(), 0);
    std::sort(sorted.begin(), sorted.end(), [annotations] (int const& a, int const& b) {
        return annotations[a].score() > annotations[b].score();
    });

    for (auto a : sorted) {
        Annotation& ann = annotations[a];
        for (int k = 0; k < numKeypoints; ++k) {
            const auto x = ann.keypoints[k*3    ];
            const auto y = ann.keypoints[k*3 + 1];
            const auto v = ann.keypoints[k*3 + 2];
            if (v == 0) { continue; }

            const auto i = std::min(std::max(0, (int)std::round(x)), w - 1);
            const auto j = std::min(std::max(0, (int)std::round(y)), h - 1);

            if (occupied[k*h*w + j*w + i] > 0.0f) {
                ann.keypoints[k*3 + 2] = 0.0f;
            } else {
                scalarSquareAddSingle(occupied.data() + k*h*w, h, w, x, y, ann.jointScales[k], 1.0f);
            }
        }
    }

    std::vector<Annotation> filtered;
    for (auto& ann : annotations) {
        for (int k = 0; k < numKeypoints; ++k) {
            if (ann.keypoints[k*3 + 2] > 0.0f) {
                filtered.push_back(ann);
                break;
            }
        }
    }
    return filtered;

    // Note: The original code sorts here on the score (descending), but
    // we sort again later on so it's a bit quicker if we skip that here.
}

void OpenPifPafPostprocessor::initTensors(int tensorWidth, int tensorHeight) {
    H = tensorHeight;
    W = tensorWidth;
    H_hr = H * (int)stride;
    W_hr = W * (int)stride;

    paf_stride_2 = H * W;
    paf_stride_1 = 4 * paf_stride_2;
    paf_stride_0 = 2 * paf_stride_1;

    pif_stride_1 = H * W;
    pif_stride_0 = 4 * pif_stride_1;

    pifhr_stride_1 = W_hr;
    pifhr_stride_0 = H_hr * pifhr_stride_1;

    indexField = makeIndexField(H, W);
    indexField_hr = makeIndexField(H_hr, W_hr);
    paf = std::vector<float>(19 * 2 * 4 * H * W);
    pif = std::vector<float>(17 * 4 * H * W);

    const int shape = C * H_hr * W_hr;
    targetsCoreOnly = std::vector<float>(shape);
    targets = std::vector<float>(shape);
    scales = std::vector<float>(shape);
    ns = std::vector<float>(shape);
}

ai_app::Object_detection::Result OpenPifPafPostprocessor::postprocess_0_8(
    int inputWidth, int inputHeight,
    int tensorWidth, int tensorHeight,
    const float* pif_c,
    const float* pif_r,
    const float* pif_s,
    const float* paf_c,
    const float* paf_r1,
    const float* paf_r2,
    const float* paf_b1,
    const float* paf_b2)
{
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;

    // Allocate the intermediate tensors the first time or when the size changes.
    if (W != tensorWidth || H != tensorHeight) {
        initTensors(tensorWidth, tensorHeight);
    }

    normalizePAF(paf_c, paf_r1, paf_r2, paf_b1, paf_b2);
    normalizePIF(pif_c, pif_r, pif_s);

    const auto ti = targetIntensities(pif);
    const auto pifhr = std::get<0>(ti);
    const auto pifhrScales = std::get<1>(ti);
    const auto pifhrCore = std::get<2>(ti);

    /*
    // For debugging
    for (int c = 0; c < 17; ++c) {
      for (int y = 0; y < H_hr; ++y) {
        for (int x = 0; x < W_hr; ++x) {
          printf("%f, ", pifhrCore[c*136*248 + y*248 + x]);
        }
      }
      printf("\n");
    }
    */

    // This returns two lists that each contain 19 tensors of shape (7, ?)
    // where the second dimension can vary in size (depends on thresholds).
    const auto pt = scorePafTarget(paf, pifhr);
    const auto pafForward = std::get<0>(pt);
    const auto pafBackward = std::get<1>(pt);

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

    auto annotations = decodeAnnotations(pifhr, pifhrScales, pifhrCore, pafForward, pafBackward);

    // Scale to input size
    const float output_stride = 8.0f;
    for (auto& ann : annotations) {
        for (int k = 0; k < numKeypoints; ++k) {
            ann.keypoints[k*3    ] *= output_stride;
            ann.keypoints[k*3 + 1] *= output_stride;
            ann.jointScales[k]     *= output_stride;
        }
    }

    // Non-maximum suppression
    if (!annotations.empty()) {
        annotations = softNMS(annotations);
    }

    // Threshold
    std::vector<Annotation> thresholded;
    for (auto& ann : annotations) {
        for (int k = 0; k < numKeypoints; ++k) {
            if (ann.keypoints[k*3 + 2] < keypointThreshold) {
                ann.keypoints[k*3 + 2] = 0.0f;
            }
        }
        if (ann.score() >= instanceThreshold) {
            thresholded.push_back(ann);
        }
    }

    std::sort(thresholded.begin(), thresholded.end(), [] (const Annotation& a, const Annotation& b) {
        return a.score() > b.score();
    });

    // Convert to normalized coordinates
    for (auto& ann : thresholded) {
        for (int k = 0; k < numKeypoints; ++k) {
            ann.keypoints[k*3    ] /= inputWidth;
            ann.keypoints[k*3 + 1] /= inputHeight;
        }
    }

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

        int minx =  std::numeric_limits<int>::max(),
            miny =  std::numeric_limits<int>::max(),
            maxx = -std::numeric_limits<int>::max(),
            maxy = -std::numeric_limits<int>::max();

        for (int k = 0; k < numKeypoints; ++k) {
            const int x = ann.keypoints[k*3    ] * 10000; // FIXME: MAGIC NUMBER.
            const int y = ann.keypoints[k*3 + 1] * 10000;
            const auto v = ann.keypoints[k*3 + 2];

            if (v > 0.0f) {
                if (x < minx) { minx = x; }
                if (x > maxx) { maxx = x; }
                if (y < miny) { miny = y; }
                if (y > maxy) { maxy = y; }
            }

            ai_app::Landmark landmark;
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
        item.bounding_box.size.x = maxx - minx;
        item.bounding_box.size.y = maxy - miny;
        item.landmarks = landmarks;

        result.items.push_back(item);
    }
    return result;
}

}