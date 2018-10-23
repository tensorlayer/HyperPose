#include "pose_detector.h"

#include <algorithm>
#include <cassert>
#include <memory>

#include "trace.hpp"
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <stdtensor>

using ttl::tensor;
using ttl::tensor_ref;

#include <openpose-plus.h>

#include "input.h"
#include "vis.h"

class pose_detector_impl : public pose_detector
{
  public:
    pose_detector_impl(const std::string &model_file,          //
                       int input_height, int input_width,      //
                       int feature_height, int feature_width,  //
                       int batch_size, bool use_f16, int gauss_kernel_size,
                       bool flip_rgb);

    void one_batch(const std::vector<std::string> &image_files, int start_idx);

    void inference(const std::vector<std::string> &image_files) override;

  private:
    const int height;
    const int width;
    const int batch_size;

    const int feature_height;
    const int feature_width;

    const bool flip_rgb;

    tensor<uint8_t, 4> hwc_images;
    tensor<float, 4> chw_images;
    tensor<float, 4> confs;
    tensor<float, 4> pafs;

    std::unique_ptr<paf_processor> process_paf;
    std::unique_ptr<pose_detection_runner> compute_feature_maps;
};

pose_detector_impl::pose_detector_impl(const std::string &model_file,      //
                                       int input_height, int input_width,  //
                                       int feature_height, int feature_width,
                                       int batch_size, bool use_f16,
                                       int gauss_kernel_size, bool flip_rgb)
    : height(input_height),
      width(input_width),
      batch_size(batch_size),
      feature_height(feature_height),
      feature_width(feature_width),
      flip_rgb(flip_rgb),
      hwc_images(batch_size, height, width, 3),
      chw_images(batch_size, 3, height, width),
      confs(batch_size, n_joins, feature_height, feature_width),
      pafs(batch_size, n_connections * 2, feature_height, feature_width),
      process_paf(create_paf_processor(feature_height, feature_width,
                                       input_height, input_width, n_joins,
                                       n_connections, gauss_kernel_size)),
      compute_feature_maps(create_pose_detection_runner(
          model_file, height, width, batch_size, use_f16))
{
}

void pose_detector_impl::one_batch(const std::vector<std::string> &image_files,
                                   int start_idx)
{
    TRACE_SCOPE(__func__);
    assert(image_files.size() <= batch_size);
    std::vector<cv::Mat> resized_images;
    {
        TRACE_SCOPE("batch read images");
        for (int i = 0; i < image_files.size(); ++i) {
            input_image(image_files[i].data(), height, width,
                        hwc_images[i].data(), chw_images[i].data(), flip_rgb);
            resized_images.push_back(cv::Mat(cv::Size(width, height), CV_8UC(3),
                                             hwc_images[i].data()));
        }
    }
    {
        TRACE_SCOPE("batch run tensorRT");
        (*compute_feature_maps)({chw_images.data()},
                                {confs.data(), pafs.data()},
                                image_files.size());
    }
    {
        TRACE_SCOPE("batch run process PAF and draw results");
        for (int i = 0; i < image_files.size(); ++i) {
            const auto humans = [&]() {
                TRACE_SCOPE("run paf_process");
                return (*process_paf)(confs[i].data(), pafs[i].data(), true);
            }();
            auto resized_image = resized_images[i];
            {
                TRACE_SCOPE("draw_results");
                std::cout << "got " << humans.size() << " humans" << std::endl;
                for (const auto &h : humans) {
                    h.print();
                    draw_human(resized_image, h);
                }
                const auto name =
                    "output" + std::to_string(start_idx + i) + ".png";
                cv::imwrite(name, resized_image);
            }
        }
    }
}

void pose_detector_impl::inference(const std::vector<std::string> &image_files)
{
    for (int i = 0; i < image_files.size(); i += batch_size) {
        std::vector<std::string> batch(
            image_files.begin() + i,
            std::min(image_files.begin() + i + batch_size, image_files.end()));
        one_batch(batch, i);
    }
}

pose_detector *create_pose_detector(const std::string &model_file,      //
                                    int input_height, int input_width,  //
                                    int feature_height, int feature_width,
                                    int batch_size, bool use_f16,
                                    int gauss_kernel_size, bool flip_rgb)
{
    return new pose_detector_impl(model_file, input_height, input_width,
                                  feature_height, feature_width, batch_size,
                                  use_f16, gauss_kernel_size, flip_rgb);
}
