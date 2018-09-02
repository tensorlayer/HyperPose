#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/examples/pose-inference/pose-detector.h>

#include "input.h"
#include "paf.h"
#include "tracer.h"
#include "vis.h"

DEFINE_string(graph_path, "", "path to freezed graph.");
DEFINE_string(input_images, "", "Comma separated list of image filenames.");

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
    std::unique_ptr<PoseDetector> detector;
    {
        tracer_t _("create_pose_detector");
        create_pose_detector(FLAGS_graph_path, detector);
    }

    for (auto f : image_files) {
        const auto img = input_image(f.c_str());
        const auto result = [&]() {
            tracer_t _("get_detection_tensors");
            return detector->get_detection_tensors(img);
        }();
        {
            tracer_t _("draw_results");
            draw_results(result);
        }
        const auto humans = estimate_paf(result);
        printf("got %lu humans\n", humans.size());
    }
}

std::vector<std::string> split(const std::string &text, const char sep)
{
    std::vector<std::string> lines;
    std::string line;
    std::istringstream ss(text);
    while (std::getline(ss, line, sep)) { lines.push_back(line); }
    return lines;
}

int main(int argc, char *argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    const auto image_files = split(FLAGS_input_images, ',');
    pose_example(image_files);
    return 0;
}
