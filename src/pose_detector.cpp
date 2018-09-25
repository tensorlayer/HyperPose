#include "pose_detector.h"

#include <algorithm>
#include <cassert>
#include <memory>

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include "input.h"
#include "paf.h"
#include "tensor.h"
#include "tracer.h"
#include "uff-runner.h"
#include "vis.h"

namespace
{
const int n_joins = 18 + 1;
const int n_connections = 17 + 2;
}  // namespace

class pose_detector_impl : public pose_detector
{
  public:
    pose_detector_impl(const std::string &model_file,          //
                       int input_height, int input_width,      //
                       int feature_height, int feature_width,  //
                       int batch_size);

    void one_batch(const std::vector<std::string> &image_files, int start_idx);

    void inference(const std::vector<std::string> &image_files) override;

  private:
    const int height;
    const int width;
    const int batch_size;

    const int feature_height;
    const int feature_width;

    tensor_t<float, 4> images;
    tensor_t<float, 4> confs;
    tensor_t<float, 4> pafs;

    std::unique_ptr<UFFRunner> runner;
    std::unique_ptr<paf_processor> paf_process;
};

pose_detector_impl::pose_detector_impl(const std::string &model_file,      //
                                       int input_height, int input_width,  //
                                       int feature_height, int feature_width,
                                       int batch_size)
    : height(input_height), width(input_width),  //
      feature_height(feature_height), feature_width(feature_width),
      batch_size(batch_size),  //
      images(nullptr, batch_size, 3, height, width),
      confs(nullptr, batch_size, n_joins, feature_height, feature_width),
      pafs(nullptr, batch_size, n_connections * 2, feature_height,
           feature_width),
      runner(create_openpose_runner(model_file, height, width, batch_size)),
      paf_process(create_paf_processor(feature_height, feature_width,  //
                                       input_height, input_width,      //
                                       n_joins, n_connections))
{
}

void pose_detector_impl::one_batch(const std::vector<std::string> &image_files,
                                   int start_idx)
{
    TRACE(__func__);
    assert(image_files.size() <= batch_size);

    std::vector<cv::Mat> resized_images;
    for (int i = 0; i < image_files.size(); ++i) {
        resized_images.push_back(
            input_image(image_files[i].c_str(), height, width,
                        images.data() + i * 3 * height * width));
    }

    {
        TRACE("batch run tensorRT");
        runner->execute({images.data()}, {confs.data(), pafs.data()},
                        image_files.size());
    }

    const int feature_size = feature_height * feature_width;
    for (int i = 0; i < image_files.size(); ++i) {
        const auto humans = [&]() {
            TRACE("run paf_process");
            return (*paf_process)(confs.data() + i * n_joins * feature_size,
                                  pafs.data() +
                                      i * 2 * n_connections * feature_size);
        }();
        auto resized_image = resized_images[i];
        {
            TRACE("draw_results");
            std::cout << "got " << humans.size() << " humans" << std::endl;
            for (const auto &h : humans) {
                h.print();
                draw_human(resized_image, h);
            }
            const auto name = "output" + std::to_string(start_idx + i) + ".png";
            cv::imwrite(name, resized_image);
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
                                    int batch_size)
{
    return new pose_detector_impl(model_file, input_height, input_width,
                                  feature_height, feature_width, batch_size);
}
