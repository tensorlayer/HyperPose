#include <hyperpose/hyperpose.hpp>

int main()
{
    using namespace hyperpose;

    const cv::Size network_resolution{ 384, 256 };
    const dnn::uff uff_model{ "../data/models/TinyVGG-V1-HW=256x384.uff", "image", { "outputs/conf", "outputs/paf" } };

    // * Input video.
    auto capture = cv::VideoCapture("../data/media/video.avi");

    // * Output video.
    auto writer = cv::VideoWriter(
        "output.avi", capture.get(cv::CAP_PROP_FOURCC), capture.get(cv::CAP_PROP_FPS), network_resolution);

    // * Create TensorRT engine.
    dnn::tensorrt engine(uff_model, network_resolution);

    // * post-processing: Using paf.
    parser::paf parser{};

    // * Create stream
    auto stream = make_stream(engine, parser);

    // * Connect input stream.
    stream.async() << capture;

    // * Connect ouput stream and wait.
    stream.sync() >> writer;
}