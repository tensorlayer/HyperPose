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

void debug(const std::string &name, const tf::Tensor &t)
{
    const auto shape = t.shape();
    std::string dims_str;
    {
        const auto dims = shape.dim_sizes();
        for (auto d : dims) { dims_str = " " + std::to_string(d) + dims_str; }
    }
    LOG(INFO) << "tensor: " << name << " :: "
              << " dtype: " << t.dtype() << " rank: " << shape.dims()
              << " dims: " << dims_str;
}

class TFPoseDetector : public PoseDetector
{
  public:
    TFPoseDetector(const std::string &graph_path);

    virtual detection_result_t
    get_detection_tensors(const detection_input_t &) override;

  private:
    static const int image_height = 368;
    static const int image_width = 432;

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

template <typename T>
std::vector<T> export4dtensor(int a, int b, int c, int d, const tf::Tensor &t)
{
    {
        tf::TensorShape shape;
        shape.AddDim(a);
        shape.AddDim(b);
        shape.AddDim(c);
        shape.AddDim(d);
        if (t.shape() != shape) {
            LOG(ERROR) << "shape miss match, want " << shape << " got "
                       << t.shape();
            exit(1);
        }
    }

    const auto &tt = t.tensor<T, 4>();

    std::vector<T> v(a * b * c * d);
    int idx = 0;
    for (int i = 0; i < a; ++i) {
        for (int j = 0; j < b; ++j) {
            for (int k = 0; k < c; ++k) {
                for (int l = 0; l < d; ++l) { v[idx++] = tt(i, j, k, l); }
            }
        }
    }
    return v;
}

PoseDetector::detection_result_t
TFPoseDetector::get_detection_tensors(const detection_input_t &input)
{
    const auto image_tensor = [](const detection_input_t &input) {
        tf::TensorShape shape;
        shape.AddDim(1);
        shape.AddDim(image_height);
        shape.AddDim(image_width);
        shape.AddDim(3);
        tf::Tensor t(tf::DT_FLOAT, shape);
        {
            using T = float;
            int idx = 0;
            for (int i = 0; i < 1; ++i) {
                for (int j = 0; j < image_height; ++j) {
                    for (int k = 0; k < image_width; ++k) {
                        for (int l = 0; l < 3; ++l) {
                            t.tensor<T, 4>()(i, j, k, l) = input[idx++];
                        }
                    }
                }
            }
        }
        return t;
    }(input);
    const auto upsample_size = []() {
        const float resize_out_ratio = 8.0;
        tf::TensorShape shape;
        shape.AddDim(2);
        tf::Tensor t(tf::DT_INT32, shape);
        using T = std::int32_t;
        t.tensor<T, 1>()(0) = T(image_height / 8 * resize_out_ratio);
        t.tensor<T, 1>()(1) = T(image_width / 8 * resize_out_ratio);
        return t;
    }();
    debug("image_tensor", image_tensor);
    debug("upsample_size", upsample_size);

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

    int idx = 0;
    for (auto &t : outputs) { debug(output_names[idx++], t); }

    using T = float;
    const int n_pos = 19;
    const auto t1 = export4dtensor<T>(1, image_height, image_width, n_pos, outputs[0]);
    const auto t2 = export4dtensor<T>(1, image_height, image_width, n_pos, outputs[1]);
    const auto t3 = export4dtensor<T>(1, image_height, image_width, n_pos * 2, outputs[2]);

    return std::make_tuple(t1, t2, t3);
}

void create_pose_detector(const std::string &model_file,
                          std::unique_ptr<PoseDetector> &p)
{
    p.reset(new TFPoseDetector(model_file));
}
