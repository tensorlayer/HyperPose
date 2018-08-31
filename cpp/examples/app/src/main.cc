#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tensorflow/examples/pose-inference/pose-detector.h"

void camera_example()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) { return; }
    cv::Mat edges;
    cv::namedWindow("edges", 1);
    for (int i = 0;; ++i) {
        printf("#%d\n", i);
        cv::Mat frame;
        cap >> frame;  // get a new frame from camera
        cv::cvtColor(frame, edges, CV_BGR2GRAY);
        cv::GaussianBlur(edges, edges, cv::Size(7, 7), 1.5, 1.5);
        cv::Canny(edges, edges, 0, 30, 3);
        cv::imshow("edges", edges);
        if (cv::waitKey(30) >= 0) { break; }
        const std::string name = "frame-" + std::to_string(i) + ".png";
        cv::imwrite(name, frame);
    }
}

void pose_example(const std::vector<std::string> &image_files)
{
    std::string graph_path = "checkpoints/freezed";
    std::unique_ptr<PoseDetector> detector;
    create_pose_detector(graph_path, detector);

    for (auto f : image_files) {
        // std::string image_path = "data/test.jpeg";
        // detector->detect_pose(image_path);
        auto results = detector->get_detection_tensors(f);
        // TODO: draw results
    }
}

int main(int argc, char *argv[])
{
    std::vector<std::string> image_files;
    for (int i = 1; i < argc; ++i) { image_files.push_back(argv[i]); }
    pose_example(image_files);
    return 0;
}
