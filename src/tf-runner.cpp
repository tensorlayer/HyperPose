#include "tf-runner.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

#include "tf-utils.h"
#include "tracer.h"

namespace tf = tensorflow;

tf::Status LoadGraph(const std::string &graph_file_name,
                     std::unique_ptr<tf::Session> &session)
{
    TRACE(__func__);

    tf::GraphDef graph_def;
    TF_RETURN_IF_ERROR(
        tf::ReadBinaryProto(tf::Env::Default(), graph_file_name, &graph_def));
    session.reset(tf::NewSession(tf::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph_def));
    return tf::Status::OK();
}

using bind_info_t = std::vector<std::pair<std::string, std::vector<int>>>;

template <typename T, typename S>
std::vector<T> get_firsts(const std::vector<std::pair<T, S>> &ps)
{
    std::vector<T> v;
    for (const auto &it : ps) { v.push_back(it.first); }
    return v;
}

class TFRunnerImpl : public TFRunner
{
  public:
    TFRunnerImpl(const std::string &graph_path, const bind_info_t &input_names,
                 const bind_info_t &output_names);

    virtual void operator()(const std::vector<void *> &input_info,
                            const std::vector<void *> &output_info) override;

  private:
    const bind_info_t input_info;
    const bind_info_t output_info;

    const std::vector<std::string> output_names;

    std::unique_ptr<tensorflow::Session> session_;
};

TFRunnerImpl::TFRunnerImpl(const std::string &graph_path,
                           const bind_info_t &input_info,
                           const bind_info_t &output_info)
    : input_info(input_info), output_info(output_info),
      output_names(get_firsts(output_info))
{
    TRACE(__func__);

    auto status = LoadGraph(graph_path, session_);
    if (!status.ok()) {
        LOG(ERROR) << status;
        exit(1);
    }
}

void TFRunnerImpl::operator()(const std::vector<void *> &inputs,
                              const std::vector<void *> &outputs)
{
    TRACE(__func__);

    std::vector<std::pair<std::string, tf::Tensor>> feed_dict;
    for (int i = 0; i < input_info.size(); ++i) {
        const auto t =
            import_tensor<float>((float *)inputs[i], input_info[i].second);
        feed_dict.push_back(std::make_pair(input_info[i].first, t));
    }

    std::vector<tf::Tensor> output_tensors;
    {
        TRACE("session->Run");
        const auto status =
            session_->Run(feed_dict, output_names, {}, &output_tensors);
        if (!status.ok()) {
            // TODO: handle error
            LOG(ERROR) << status;
            exit(1);
        }
    }

    for (int i = 0; i < output_info.size(); ++i) {
        export_tensor(output_tensors[i], (float *)outputs[i]);
    }
}

void create_openpose_runner(const std::string &model_file, const int height,
                            const int width, std::unique_ptr<TFRunner> &p)
{
    TRACE(__func__);

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
    p.reset(new TFRunnerImpl(model_file, input_names, output_names));
}
