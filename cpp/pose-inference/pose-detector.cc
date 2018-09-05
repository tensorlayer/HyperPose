#include "tensorflow/examples/pose-inference/pose-detector.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

tf::Status LoadGraph(const std::string &graph_file_name,
                     std::unique_ptr<tf::Session> &session)
{
    tf::GraphDef graph_def;
    TF_RETURN_IF_ERROR(
        tf::ReadBinaryProto(tf::Env::Default(), graph_file_name, &graph_def));
    session.reset(tf::NewSession(tf::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph_def));
    return tf::Status::OK();
}

class TFPoseDetector : public PoseDetector
{
  public:
    TFPoseDetector(const std::string &graph_path);

    virtual output_t get_detection_tensors(const input_t &) override;

  private:
    const std::vector<std::string> input_names = {
        "image",
        "upsample_size",
    };

    const std::vector<std::string> output_names = {
        "upsample_heatmat",
        "tensor_peaks",
        "upsample_pafmat",
    };

    std::unique_ptr<tensorflow::Session> session_;
};

TFPoseDetector::TFPoseDetector(const std::string &graph_path)
{
    auto status = LoadGraph(graph_path, session_);
    if (!status.ok()) {
        LOG(ERROR) << status;
        exit(1);
    }
}

template <typename T> tf::Tensor import4dtensor(const tensor_t<T, 4> &input)
{
    // const auto [a, b, c, d] = input.dims; // requires C++17
    const int a = input.dims[0];
    const int b = input.dims[1];
    const int c = input.dims[2];
    const int d = input.dims[3];

    // TODO: infer type from T
    tf::Tensor t(tf::DT_FLOAT, tf::TensorShape({a, b, c, d}));
    {
        int idx = 0;
        for (int i = 0; i < a; ++i) {
            for (int j = 0; j < b; ++j) {
                for (int k = 0; k < c; ++k) {
                    for (int l = 0; l < d; ++l) {
                        t.tensor<T, 4>()(i, j, k, l) = input.data[idx++];
                    }
                }
            }
        }
    }
    return t;
}

template <typename T> tensor_t<float, 4> export4dtensor(const tf::Tensor &t)
{
    tensor_t<float, 4> output;
    for (int i = 0; i < 4; ++i) { output.dims[i] = t.shape().dim_size(i); }

    // const auto [a, b, c, d] = output.dims; // requires C++17
    const int a = output.dims[0];
    const int b = output.dims[1];
    const int c = output.dims[2];
    const int d = output.dims[3];

    output.data.resize(a * b * c * d);
    const auto &tt = t.tensor<T, 4>();
    {
        int idx = 0;
        for (int i = 0; i < a; ++i) {
            for (int j = 0; j < b; ++j) {
                for (int k = 0; k < c; ++k) {
                    for (int l = 0; l < d; ++l) {
                        output.data[idx++] = tt(i, j, k, l);
                    }
                }
            }
        }
    }
    return output;
}

PoseDetector::output_t
TFPoseDetector::get_detection_tensors(const input_t &input)
{
    const auto image_tensor = import4dtensor(input);
    const auto upsample_size = [&]() {
        // const auto [_n, height, width, _c] = input.dims; // requires C++ 17
        const int height = input.dims[1];
        const int width = input.dims[2];

        const float resize_out_ratio = 8.0;
        const auto f = [&](int size) {
            return std::int32_t(size / 8 * resize_out_ratio);
        };
        tf::Tensor t(tf::DT_INT32, tf::TensorShape({2}));
        t.tensor<std::int32_t, 1>()(0) = f(height);
        t.tensor<std::int32_t, 1>()(1) = f(width);
        return t;
    }();

    std::vector<tf::Tensor> outputs;
    {
        auto status = session_->Run(
            {
                {input_names[0], image_tensor},
                {input_names[1], upsample_size},
            },
            output_names, {}, &outputs);
        if (!status.ok()) {
            // TODO: handle error
            LOG(ERROR) << status;
            exit(1);
        }
    }

    using T = float;
    return std::make_tuple(export4dtensor<T>(outputs[0]),
                           export4dtensor<T>(outputs[1]),
                           export4dtensor<T>(outputs[2]));
}

void create_pose_detector(const std::string &model_file,
                          std::unique_ptr<PoseDetector> &p)
{
    p.reset(new TFPoseDetector(model_file));
}
