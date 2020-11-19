#pragma once

#include <cstdio>
#include <string>
#include <vector>
#include <set>

#include "object_detection.hpp"

namespace lpdnn::aiapp_impl {

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
class OpenPifPafPostprocessor
{
public:
  OpenPifPafPostprocessor() : H(0), W(0) { }

  /**
    Applies post-processing to OpenPifPaf output.

    \param inpWidth Width of the input tensor in pixels.
    \param inpHeight Height of the input tensor in pixels.
    \param tensorWidth Width of the neural network's PIF and PAF outputs.
    \param tensorHeight Height of the neural network's PIF and PAF outputs.
  */
  ai_app::Object_detection::Result postprocess_0_8(
    int inpWidth, int inpHeight, int tensorWidth, int tensorHeight,
    const float* pif_c,  // 17xHxW
    const float* pif_r,  // 34xHxW
    const float* pif_s,  // 17xHxW
    const float* paf_c,  // 19xHxW
    const float* paf_r1, // 38xHxW
    const float* paf_r2, // 38xHxW
    const float* paf_b1, // 19xHxW
    const float* paf_b2  // 19xHxW
  );

public:
  static const int numKeypoints = 17;
  static const int numBones = 19;

  // Connections between the different keypoint indices.
  // Note: these start at 1, not 0!
  static const int bones[19][2];

private:
  struct Annotation {
    // Array of `numKeypoints * 3` elements:
    // - element `i*3 + 0` is x-coordinate (normalized)
    // - element `i*3 + 1` is y-coordinate (normalized)
    // - element `i*3 + 2` is confidence score
    std::vector<float> keypoints;

    std::vector<float> jointScales;

    Annotation(int j, float x, float y, float v) : keypoints(numKeypoints * 3),
                                                   jointScales(numKeypoints)
    {
      keypoints[j*3    ] = x;
      keypoints[j*3 + 1] = y;
      keypoints[j*3 + 2] = v;
    }

    /**
      Overall confidence score for the entire skeleton.
    */
    [[nodiscard]] float score() const {
      float maxv = 0.0f;
      float vv = 0.0f;
      for (int k = 0; k < numKeypoints; ++k) {
        auto v = keypoints[k*3 + 2];
        maxv = std::max(maxv, v);
        vv += v * v;
      }
      return 0.1f * maxv + 0.9f * vv / (float)numKeypoints;
    }
  };

  // 0: confidence of origin
  // 1: connection index
  // 2: forward?
  // 3: joint index 1 (not corrected for forward)
  // 4: joint index 2 (not corrected for forward)
  typedef std::tuple<float, int, bool, int, int> frontier_t;
  typedef std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> Target_intensity;
  typedef std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>> Paf_target;
  typedef std::tuple<float, int, float, float> Pifhr_seed;
  typedef std::tuple<float, float, float> Connection;

private:
  void initTensors(int tensorWidth, int tensorHeight);

  void normalizePAF(const float* intensityFields,
                    const float* j1Fields,
                    const float* j2Fields,
                    const float* j1FieldsLogb,
                    const float* j2FieldsLogb);

  void normalizePIF(const float* jointIntensityFields,
                    const float* jointFields,
                    const float* scaleFields);

  Target_intensity
  targetIntensities(const std::vector<float>& pif,
                    float v_th = 0.1f,
                    bool coreOnly = false);

  Paf_target
  scorePafTarget(const std::vector<float>& pafvec,
                 const std::vector<float>& pifhr,
                 float pifhr_floor = 0.01f,
                 float score_th = 0.1f) const;

  std::vector<Pifhr_seed>
  pifhrSeeds(const std::vector<float>& pifhrScales,
             const std::vector<float>& pifhrCore);

  static std::vector<float> pafCenter(const std::vector<float>& paf_field,
                               float x, float y, float sigma = 1.0f);

  static Connection
  growConnection(float x, float y, const std::vector<float>& paf_field_);

  static std::vector<frontier_t> frontier(Annotation& ann);

  frontier_t frontierIter(Annotation& ann);

  void grow(Annotation& ann,
            const std::vector<std::vector<float>>& pafForward,
            const std::vector<std::vector<float>>& pafBackward,
            float th = 0.1f);

  void fillJointScales(Annotation& ann,
                       const std::vector<float>& scales,
                       int fieldH,
                       int fieldW,
                       float hr_scale);

  std::vector<Annotation>
  decodeAnnotations(const std::vector<float>& pifhr,
                    const std::vector<float>& pifhrScales,
                    const std::vector<float>& pifhrCore,
                    const std::vector<std::vector<float>>& pafForward,
                    const std::vector<std::vector<float>>& pafBackward);

  std::vector<Annotation> softNMS(std::vector<Annotation>& annotations);

private:
  // Used to normalize the skeleton keypoint coordinates to [0, 1].
  float inputWidth, inputHeight;

  // Tensor dimensions (hr = high-resolution).
  int H, W, H_hr, W_hr;

  // Strides for tensor dimensions.
  size_t paf_stride_2, paf_stride_1, paf_stride_0;
  size_t pif_stride_1, pif_stride_0;
  size_t pifhr_stride_1, pifhr_stride_0;

  // Temporary tensors.
  std::vector<float> indexField;     // 2 x H x W
  std::vector<float> indexField_hr;  // 2 x H x W
  std::vector<float> paf;            // 19 x 2 x 4 x H x W
  std::vector<float> pif;            // 17 x     4 x H x W

  // Filled in by targetIntensities().
  std::vector<float> targetsCoreOnly;
  std::vector<float> targets;
  std::vector<float> scales;
  std::vector<float> ns;

  std::set<std::tuple<int, bool>> blockFrontier;
  bool frontierActive;
};

}

