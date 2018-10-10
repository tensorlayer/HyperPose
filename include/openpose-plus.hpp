// The public C++ API of openpose-plus
#pragma once
#include <openpose-plus/human.h>

/*! \interface pose_detection_runner
    A class that runs a the pose detection model, which computes the feature
   maps from input image.
*/
class pose_detection_runner
{
  public:
    //! Run the pose detection model.
    virtual void
    operator()(const std::vector<void *>
                   &inputs /*! should contain 1 pointer to a float array of
                              length max_batch_size * 3 * H * W */
               ,
               const std::vector<void *>
                   &outputs /*! should container 2 pointers to heatmap and PAF
                               map, heatmap :: max_batch_size * 19 * H' * W',
                               PAF map :: max_batch_size * (2*19) * H' * W' */
               ,
               int batchSize = 1 /*! number of batches */) = 0;

    virtual ~pose_detection_runner() {}
};

//! Creates a pose_detection_runner.
// Currently TensorRT is used as backend.
// TODO: support TensorFlow backend
pose_detection_runner *create_pose_detection_runner(
    const std::string &model_file /*! path to the exported uff model file */,
    int input_height /*! height of the input image */,
    int input_width /*! width of the input image */,
    int max_batch_size /*! max batch size */,
    bool use_f16 /*! if use float 16 */);

/*! \interface paf_processor
    A class that process the feature maps of a pose detection model, i.e
confidence map and PAF.
*/
class paf_processor
{
  public:
    //! Generate humans from feature maps produced by the openpose model
    virtual std::vector<human_t> operator()(const float * /*! heatmap */,
                                            const float * /*! PAF map */,
                                            bool /*! use GPU */) = 0;

    virtual ~paf_processor() {}
};

//! Create a paf_processor.
paf_processor *
create_paf_processor(int input_height /*! height of feature maps */,
                     int input_width /*! width of feature maps */,
                     int height /*! height of output, usually equal to input
                                   image height */
                     ,
                     int width /*! width of output, usually equal to input image
                                  width */
                     ,
                     int n_joins /*! must be 19 for now */,
                     int n_connections /*! must be 19 for now */, int gauss_kernel_size /*! gauss kernel size for smooth the feature maps after resize */);
