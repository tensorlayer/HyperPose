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

    const std::string input_name = "image";
    const std::vector<std::string> output_names = {
        "model/cpm/stage6/branch1/conf/BiasAdd",
        "model/cpm/stage6/branch2/pafs/BiasAdd",
        "Select",
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

template <typename T, typename Tensor>
std::vector<T> export4dtensor(int a, int b, int c, int d, const Tensor &tt)
{
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

template <typename T> std::vector<T> export_conf_tensor(const tf::Tensor &t)
{
    auto tt = t.tensor<T, 4>();
    return export4dtensor<T>(1, 46, 54, 19, tt);
}

template <typename T> std::vector<T> export_pafs_tensor(const tf::Tensor &t)
{
    auto tt = t.tensor<T, 4>();
    return export4dtensor<T>(1, 46, 54, 38, tt);
}

template <typename T> std::vector<T> export_peek_tensor(const tf::Tensor &t)
{
    auto tt = t.tensor<T, 4>();
    return export4dtensor<T>(1, 46, 54, 38, tt);
}

PoseDetector::detection_result_t
TFPoseDetector::get_detection_tensors(const detection_input_t &input)
{
    tf::TensorShape shape;
    shape.AddDim(1);
    shape.AddDim(image_height);
    shape.AddDim(image_width);
    shape.AddDim(3);
    tf::Tensor resized_tensor(tf::DT_FLOAT, shape);
    {
        using T = float;
        int idx = 0;
        for (int i = 0; i < 1; ++i) {
            for (int j = 0; j < image_height; ++j) {
                for (int k = 0; k < image_width; ++k) {
                    for (int l = 0; l < 3; ++l) {
                        resized_tensor.tensor<T, 4>()(i, j, k, l) =
                            input[idx++];
                    }
                }
            }
        }
    }
    debug("resized_tensor", resized_tensor);

    std::vector<tf::Tensor> outputs;
    {
        auto status = session_->Run({{input_name, resized_tensor}},
                                    output_names, {}, &outputs);
        if (!status.ok()) {
            // TODO: handle error
            LOG(ERROR) << status;
            exit(1);
        }
    }

    using T = float;
    const auto t1 = export_conf_tensor<T>(outputs[0]);
    const auto t2 = export_pafs_tensor<T>(outputs[1]);
    const auto t3 = export_peek_tensor<T>(outputs[2]);

    return std::make_tuple(t1, t2, t3);
}

void create_pose_detector(const std::string &model_file,
                          std::unique_ptr<PoseDetector> &p)
{
    p.reset(new TFPoseDetector(model_file));
}
