#include <memory>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <tensorflow/examples/pose-inference/pose-detector.h>

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
    static const int image_height = 368;
    static const int image_width = 432;

    std::unique_ptr<PoseDetector> detector;
    {
        tracer_t _("create_pose_detector");
        create_pose_detector(FLAGS_graph_path, detector);
    }

    {
        int idx = 0;
        for (auto f : image_files) {
            tracer_t _("handle_image");

            const auto img = cv::imread(f);
            const cv::Size new_size(image_width, image_height);
            cv::Mat resized_image(new_size, CV_8UC(3));
            cv::resize(img, resized_image, resized_image.size(), 0, 0);

            PoseDetector::detection_input_t input;
            {
                input.dims =
                    std::array<int, 4>({1, image_height, image_width, 3});
                input.data.resize(1 * image_height * image_width * 3);
                {
                    int idx = 0;
                    for (int i = 0; i < image_height; ++i) {
                        for (int j = 0; j < image_width; ++j) {
                            const auto pix = resized_image.at<cv::Vec3b>(i, j);
                            input.data[idx++] = pix[0] / 255.0;
                            input.data[idx++] = pix[1] / 255.0;
                            input.data[idx++] = pix[2] / 255.0;
                        }
                    }
                }
            }

            const auto results = [&]() {
                tracer_t _("get_detection_tensors");
                return detector->get_detection_tensors(input);
            }();

            const auto humanss = estimate_paf(results);
            const auto humans = humanss[0];

            printf("got %lu none-empty humans\n", humans.size());
            for (const auto &h : humans) {
                h.print();
                draw_human(resized_image, h);
            }

            // cv::imshow("original", dst);
            const auto name = "output" + std::to_string(++idx) + ".png";
            cv::imwrite(name, resized_image);
        }
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
