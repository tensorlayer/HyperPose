#pragma once

/// \file model.hpp
/// \brief The configurations of model files.

#include <string>
#include <vector>

namespace hyperpose {

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

    /// \brief The configuration struct for the serialized TensorRT(ICudaEngine) model file.
    /// \note By compiling your model into serialized format, it can reduce the initializing time of loading and compiling models
    /// from other format(e.g., Uff, ONNX). And this is the recommended way to execute a model.
    struct tensorrt_serialized {
        std::string model_path; ///< Path to the serialized TensorRT(ICudaEngine) file.
    };
}
}