#include "tensorflow/examples/pose-inference/pose-detector.h"

#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

tf::Status ReadEntireFile(tf::Env *env, const std::string &filename,
                          tf::Tensor *output)
{
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    std::string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                            "' expected ", file_size, " got ",
                                            data.size());
    }
    output->scalar<std::string>()() = data.ToString();
    return tf::Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
tf::Status ReadTensorFromImageFile(const std::string &file_name,
                                   const int input_height,
                                   const int input_width,
                                   const float input_mean,
                                   const float input_std,
                                   std::vector<tf::Tensor> *out_tensors)
{
    auto root = tensorflow::Scope::NewRootScope();

    std::string input_name = "file_reader";
    std::string output_name = "normalized";

    // read file_name into a tensor named input
    tf::Tensor input(tf::DT_STRING, tf::TensorShape());
    TF_RETURN_IF_ERROR(ReadEntireFile(tf::Env::Default(), file_name, &input));

    // use a placeholder to read input data
    auto file_reader =
        tf::ops::Placeholder(root.WithOpName("input"), tf::DataType::DT_STRING);

    std::vector<std::pair<std::string, tf::Tensor>> inputs = {
        {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tf::Output image_reader;
    if (tf::str_util::EndsWith(file_name, ".png")) {
        image_reader =
            tf::ops::DecodePng(root.WithOpName("png_reader"), file_reader,
                               tf::ops::DecodePng::Channels(wanted_channels));
    } else if (tf::str_util::EndsWith(file_name, ".gif")) {
        // gif decoder returns 4-D tensor, remove the first dim
        image_reader = tf::ops::Squeeze(
            root.WithOpName("squeeze_first_dim"),
            tf::ops::DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else if (tf::str_util::EndsWith(file_name, ".bmp")) {
        image_reader =
            tf::ops::DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
    } else {
        // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
        image_reader =
            tf::ops::DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                tf::ops::DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    auto float_caster = tf::ops::Cast(root.WithOpName("float_caster"),
                                      image_reader, tensorflow::DT_FLOAT);
    // The convention for image ops in TensorFlow is that all images are
    // expected to be in batches, so that they're four-dimensional arrays with
    // indices of [batch, height, width, channel]. Because we only have a single
    // image, we have to add a batch dimension of 1 to the start with
    // ExpandDims().
    auto dims_expander = tf::ops::ExpandDims(root, float_caster, 0);
    // Bilinearly resize the image to fit the required dimensions.
    auto resized = tf::ops::ResizeBilinear(
        root, dims_expander,
        tf::ops::Const(root.WithOpName("size"), {input_height, input_width}));
    // Subtract the mean and divide by the scale.
    tf::ops::Div(root.WithOpName(output_name),
                 tf::ops::Sub(root, resized, {input_mean}), {input_std});

    // This runs the GraphDef network definition that we've just constructed,
    // and returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
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

tf::Status read_and_resize_image(const std::string &image_path,
                                 int image_height, int image_width,
                                 std::vector<tf::Tensor> &resized_tensors)
{
    const float input_mean = 0;
    const float input_std = 255;
    auto status =
        ReadTensorFromImageFile(image_path, image_height, image_width,
                                input_mean, input_std, &resized_tensors);
    TF_RETURN_IF_ERROR(status);
    return tf::Status::OK();
}

class TFPoseDetector : public PoseDetector
{
  public:
    TFPoseDetector(const std::string &graph_path);
    void detect_pose(const std::string &image_path) override;

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

    tensorflow::Status detect_pose_(const std::string &image_path);

    tensorflow::Status
    LoadGraph_(const std::string &graph_file_name,
               std::unique_ptr<tensorflow::Session> &session);
};

TFPoseDetector::TFPoseDetector(const std::string &graph_path)
{
    auto status = LoadGraph_(graph_path, session_);
    if (!status.ok()) { exit(1); }
}

void TFPoseDetector::detect_pose(const std::string &image_path)
{
    auto status = detect_pose_(image_path);
    if (!status.ok()) { LOG(WARNING) << status; }
}

template <typename T> void for4d(int a, int b, int c, int d, const T &tt)
{
    for (int i = 0; i < a; ++i) {
        for (int j = 0; j < b; ++j) {
            for (int k = 0; k < c; ++k) {
                for (int l = 0; l < d; ++l) {
                    std::cout << "T[" << i << "," << j << "," << k << "," << l
                              << "] = " << tt(i, j, k, l) << std::endl;
                }
            }
        }
    }
}

void export_conf_tensor(const tf::Tensor &t)
{
    auto tt = t.tensor<float, 4>();
    // for4d(1, 46, 54, 19, tt);
}

void export_pafs_tensor(const tf::Tensor &t)
{
    auto tt = t.tensor<float, 4>();
    // for4d(1, 46, 54, 38, tt);
}

void export_peek_tensor(const tf::Tensor &t)
{
    auto tt = t.tensor<float, 4>();
    // for4d(1, 46, 54, 38, tt);
}

tf::Status TFPoseDetector::detect_pose_(const std::string &image_path)
{
    // TODO: read image with opencv
    std::vector<tf::Tensor> resized_tensors;
    TF_RETURN_IF_ERROR(read_and_resize_image(image_path, image_height,
                                             image_width, resized_tensors));
    const auto &resized_tensor = resized_tensors[0];

    std::vector<tf::Tensor> outputs;
    auto status = session_->Run({{input_name, resized_tensor}}, output_names,
                                {}, &outputs);
    TF_RETURN_IF_ERROR(status);

    LOG(INFO) << "got " << outputs.size() << " outputs";
    int idx = 0;
    for (auto &t : outputs) { debug("output" + std::to_string(++idx), t); }

    export_conf_tensor(outputs[0]);
    export_pafs_tensor(outputs[1]);
    export_peek_tensor(outputs[2]);

    // TODO: run pafprocess

    return tf::Status::OK();
}

tf::Status TFPoseDetector::LoadGraph_(const std::string &graph_file_name,
                                      std::unique_ptr<tf::Session> &session)
{
    tf::GraphDef graph_def;
    TF_RETURN_IF_ERROR(
        tf::ReadBinaryProto(tf::Env::Default(), graph_file_name, &graph_def));
    session.reset(tf::NewSession(tf::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph_def));
    return tf::Status::OK();
}

void create_pose_detector(std::unique_ptr<PoseDetector> &p)
{
    std::string graph_path = "../checkpoints/freezed";
    p.reset(new TFPoseDetector(graph_path));
}
