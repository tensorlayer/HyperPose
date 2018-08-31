#include <memory>
#include <string>
#include <vector>

#include "tensorflow/examples/pose-inference/pose-detector.h"

int main()
{
    std::unique_ptr<PoseDetector> detector;
    create_pose_detector(detector);

    const int n = 1;
    for (int i = 0; i < n; ++i) {
        std::string image_path = "../data/test.jpeg";
        detector->detect_pose(image_path);
    }
    return 0;
}
