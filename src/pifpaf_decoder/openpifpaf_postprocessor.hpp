#pragma once

#include <set>
#include <stdio.h>
#include <string>
#include <vector>

#include "object_detection.hpp"

namespace lpdnn {
namespace aiapp_impl {

    using FBContainer = std::array<std::array<std::vector<float>, 9>, 19>;

    /**
  Post-processing logic for OpenPifPaf

  \note This object caches the big tensors to save on memory allocations.
  This means it's best to make one instance of this class and keep using it.
  For the most efficient results, make sure the input tensors are always the
  same width and height.

  \note This code is not threadsafe. Don't call it from multiple threads at
  the same time. If you must use multiple threads, give each thread its own
  instance of this class.
 */
    class OpenPifPafPostprocessor {
    public:
        OpenPifPafPostprocessor()
            : H(0)
            , W(0)
        {
        }

    public:
        static constexpr int N_PIFPAF_KEYPOINTS = 17;
        static constexpr int N_PIFPAF_BONES = 19;

        // Connections between the different keypoint indices.
        // Note: these start at 1, not 0!
        static const int bones[19][2];
        float keypointThreshold;

        ai_app::Object_detection::Result postprocess(
            int inputWidth, int inputHeight,
            int tensorWidth, int tensorHeight,
            const std::vector<float>& pif,
            const std::vector<float>& paf);

    private:
        struct Annotation {
            // Array of `N_PIFPAF_KEYPOINTS * 3` elements:
            // - element `i*3 + 0` is x-coordinate (normalized)
            // - element `i*3 + 1` is y-coordinate (normalized)
            // - element `i*3 + 2` is confidence score
            std::vector<float> keypoints;

            std::vector<float> jointScales;

            Annotation(int j, float x, float y, float v)
                : keypoints(N_PIFPAF_KEYPOINTS * 3)
                , jointScales(N_PIFPAF_KEYPOINTS)
            {
                keypoints[j * 3] = x;
                keypoints[j * 3 + 1] = y;
                keypoints[j * 3 + 2] = v;
            }

            /**
      Overall confidence score for the entire skeleton.
    */
            float score() const
            {
                float maxv = 0.0f;
                float vv = 0.0f;
                for (int k = 0; k < N_PIFPAF_KEYPOINTS; ++k) {
                    auto v = keypoints[k * 3 + 2];
                    if (v > maxv) {
                        maxv = v;
                    }
                    vv += v * v;
                }
                return 0.1f * maxv + 0.9f * vv / (float)N_PIFPAF_KEYPOINTS;
            }
        };

        typedef std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> Target_intensity;

    private:
        void initTensors(int tensorWidth, int tensorHeight);

        Target_intensity
        targetIntensities(const std::vector<float>& pif,
            float v_th = 0.1f,
            bool coreOnly = false);

        std::tuple<float, float, float, float>
        growConnectionBlend(float x, float y, float s, const std::array<std::vector<float>, 9>& paf_field_);

        // frontier_t frontierIter(Annotation& ann);

        void grow(Annotation& ann,
            const FBContainer& pafForward,
            const FBContainer& pafBackward);

        std::vector<Annotation> softNMS(std::vector<Annotation>& annotations);

    private:
        // Tensor dimensions (hr = high-resolution).
        int H, W, H_hr, W_hr;

        // Strides for tensor dimensions.
        size_t pif_stride_1, pif_stride_0;
        size_t pifhr_stride_1, pifhr_stride_0;

        // Filled in by targetIntensities().
        std::vector<float> targetsCoreOnly;
        std::vector<float> targets;
        std::vector<float> scales;
        std::vector<float> ns;
    };

}
}
