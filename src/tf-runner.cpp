#include "tf-runner.h"
#include "tf-utils.h"

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>

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

using bind_info_t = std::map<std::string, std::vector<int>>;

class OpenposeRunnerImpl : public TFRunner
{
  public:
    OpenposeRunnerImpl(const std::string &graph_path,
                       const bind_info_t &input_names,
                       const bind_info_t &output_names);

    virtual void operator()(const std::vector<void *> &inputs,
                            const std::vector<void *> &outputs) override;

  private:
    const bind_info_t input_names;
    const bind_info_t output_names;
    std::unique_ptr<tensorflow::Session> session_;
};

OpenposeRunnerImpl::OpenposeRunnerImpl(const std::string &graph_path,
                                       const bind_info_t &input_names,
                                       const bind_info_t &output_names)
    : input_names(input_names), output_names(output_names)
{
    auto status = LoadGraph(graph_path, session_);
    if (!status.ok()) {
        LOG(ERROR) << status;
        exit(1);
    }
}

void OpenposeRunnerImpl::operator()(const std::vector<void *> &inputs,
                                    const std::vector<void *> &outputs)
{
    // const auto image_tensor = import4dtensor(input);
    // const auto upsample_size = [&]() {
    //     // const auto [_n, height, width, _c] = input.dims; // requires C++
    //     17 const int height = input.dims[1]; const int width = input.dims[2];

    //     const float resize_out_ratio = 8.0;
    //     const auto f = [&](int size) {
    //         return std::int32_t(size / 8 * resize_out_ratio);
    //     };
    //     tf::Tensor t(tf::DT_INT32, tf::TensorShape({2}));
    //     t.tensor<std::int32_t, 1>()(0) = f(height);
    //     t.tensor<std::int32_t, 1>()(1) = f(width);
    //     return t;
    // }();

    // std::vector<tf::Tensor> outputs;
    // {
    //     auto status = session_->Run(
    //         {
    //             {input_names[0], image_tensor},
    //             {input_names[1], upsample_size},
    //         },
    //         output_names, {}, &outputs);
    //     if (!status.ok()) {
    //         // TODO: handle error
    //         LOG(ERROR) << status;
    //         exit(1);
    //     }
    // }

    // using T = float;
    // return std::make_tuple(export4dtensor<T>(outputs[0]),
    //                        export4dtensor<T>(outputs[1]),
    //                        export4dtensor<T>(outputs[2]));

    //
}

void create_openpose_runner(const std::string &model_file, const int height,
                            const int width, std::unique_ptr<TFRunner> &p)
{
    const int n_joins = 18 + 1;
    const int n_connections = 17 + 2;

    const int f_height = height / 8;
    const int f_width = width / 8;

    const bind_info_t input_names = {
        {"image", {height, width, 3}},
    };

    const bind_info_t output_names = {
        {"outputs/conf", {f_height, f_width, n_joins}},
        {"outputs/paf", {f_height, f_width, 2 * n_connections}},
    };
    p.reset(new OpenposeRunnerImpl(model_file, input_names, output_names));
}
