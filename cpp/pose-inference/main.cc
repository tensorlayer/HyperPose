#include <memory>
#include <string>
#include <vector>

#include "tensorflow/examples/pose-inference/pose-detector.h"

int main()
{
    // TODO: make it a flag
    std::string graph_path = "../checkpoints/freezed";

    std::unique_ptr<PoseDetector> detector;
    create_pose_detector(graph_path, detector);

    const int n = 1;
    for (int i = 0; i < n; ++i) {
        // TODO: make it a flag
        std::string image_path = "../data/test.jpeg";
        // detector->detect_pose(image_path);
        auto results = detector->get_detection_tensors(image_path);
        // TODO: draw results
    }
    return 0;
}
