#pragma once

/// \file model.hpp
/// \brief The configurations of model files.

#include <string>
#include <vector>

namespace poseplus {

namespace dnn {

    /// \brief The configuration struct for Uff models.
    /// \note ONNX model file is recommended as it supports more operators than Uff and doesn't require users to
    /// set binding names.
    struct uff {
        std::string model_path; ///< Path to the [Uff model]((https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/uff/uff.html)) file.
        std::string input_name; ///< The name of input node. (for simplicity, we only consider 1 input node model)
        std::vector<std::string> output_names; ///< The names of output nodes. (e.g., openpose model will generate "paf" and "conf")
    };

    /// \brief The configuration struct for ONNX models.
    struct onnx {
        std::string model_path; ///< Path to the [ONNX model](https://onnx.ai/) file.
    };
}
}