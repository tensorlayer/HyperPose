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
        "output_video.avi", capture.get(cv::CAP_PROP_FOURCC), capture.get(cv::CAP_PROP_FPS), network_resolution);

    // * Create TensorRT engine.
    dnn::tensorrt engine(uff_model, network_resolution);

    // * post-processing: Using paf.
    parser::paf parser{};

    while (capture.isOpened()) {
        std::vector<cv::Mat> batch;
        for (int i = 0; i < engine.max_batch_size(); ++i) {
            cv::Mat mat;
            capture >> mat;
            if (mat.empty())
                break;
            batch.push_back(mat);
        }

        if (batch.empty())
            break;

        // * TensorRT Inference.
        auto feature_map_packets = engine.inference(batch);

        // * Paf.
        std::vector<std::vector<human_t>> pose_vectors;
        pose_vectors.reserve(feature_map_packets.size());
        for (auto&& packet : feature_map_packets)
            pose_vectors.push_back(parser.process(packet[0], packet[1]));

        // * Visualization
        for (size_t i = 0; i < batch.size(); ++i) {
            cv::resize(batch[i], batch[i], network_resolution);
            for (auto&& pose : pose_vectors[i])
                draw_human(batch[i], pose);
            writer << batch[i];
        }
    }
}