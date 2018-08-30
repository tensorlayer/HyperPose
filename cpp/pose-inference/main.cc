#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/public/session.h"

namespace tf = tensorflow;

const int image_height = 368;
const int image_width = 432;

tf::Status LoadGraph(const std::string &graph_file_name,
                     std::unique_ptr<tf::Session> &session)
{
    tf::GraphDef graph_def;
    {
        auto status = tf::ReadBinaryProto(tf::Env::Default(), graph_file_name,
                                          &graph_def);
        if (!status.ok()) { return status; }
    }
    {
        session.reset(tf::NewSession(tf::SessionOptions()));
        auto status = session->Create(graph_def);
        if (!status.ok()) { return status; }
    }
    return tf::Status::OK();
}

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

tf::Status createTestTensor(tf::Tensor *output)
{
    // const int width = 28;
    // const int height = 28;

    // auto &t =
    // output->tensor<float, 2>()() = 1;//{{1, 2, 3}};

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
    // using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

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

tf::Status openpose()
{
    std::unique_ptr<tf::Session> session;
    std::string graph_path = "../checkpoints/freezed";
    TF_RETURN_IF_ERROR(LoadGraph(graph_path, session));

    {
        const std::string input_name = "image";
        const std::vector<std::string> output_names = {
            "model/cpm/stage6/branch1/conf/BiasAdd",
            "model/cpm/stage6/branch2/pafs/BiasAdd",
            "Select",
        };

        tf::TensorShape shape;
        shape.AddDim(1);
        shape.AddDim(image_height);
        shape.AddDim(image_width);
        shape.AddDim(3);

        tf::Tensor resized_tensor(tf::DT_FLOAT, shape);

        std::vector<tf::Tensor> outputs;
        auto status = session->Run({{input_name, resized_tensor}}, output_names,
                                   {}, &outputs);
        TF_RETURN_IF_ERROR(status);

        LOG(INFO) << "got " << outputs.size() << " outputs";
        // for (auto &o : outputs) { LOG(INFO) << o.tensor<float, 2>(); }
    }

    return tf::Status::OK();
}

int main()
{
    auto status = example_2();
    if (!status.ok()) {
        LOG(ERROR) << status;
        exit(1);
    }
    return 0;
}
