#pragma once

/// \file paf.hpp
/// \brief Post-processing using Part Affinity Field (PAF).
/// \author Jiawei Liu(github.com/ganler)

#include "../../utility/data.hpp"

namespace poseplus {

/// \note In OpenPose-Plus, the pose estimation pipeline consists of DNN inference and parsing(post-processing). The
/// parser part implementation is under the namespace `poseplus::parser`.
namespace parser {

    /// \brief Post-processing using Part Affinity Field (PAF).
    /// \see https://arxiv.org/abs/1812.08008
    class paf {
    public:
        /// \brief Constructor indicating the image size and thresholds.
        ///
        /// \param resolution_size The size(width, height) of expected resolution for the post-processing.
        /// \param paf_thresh The threshold of Part Affinity Field.
        /// \param conf_thresh The activation threshold.
        /// \note Before doing PAF, the (width, height) of feature map will be expanded to `resolution_size` to perform
        /// a more accurate post processing.
        paf(cv::Size resolution_size, float paf_thresh = 0.05, float conf_thresh = 0.05);

        /// \brief Function to process one image.
        ///
        /// \code
        /// // Initialization of PAF.
        /// poseplus::parser::paf paf_processor(/* image size */);
        ///
        /// // ...
        ///
        /// // Do inference.
        /// // Note that, tensor_pairs.size is equal to the batch_size process. Each of them represents one image.
        /// auto tensor_pairs = engine.inference(...);
        ///
        /// for(auto&& tensor_pair/* PAF, CONF */ : tensor_pairs)
        ///    human_topology = paf_processor.process(tensor_pair[0], tensor_pair[1]);
        /// \endcode
        ///
        /// \param paf The paf tensor.
        /// \param conf The conf tensor.
        /// \return All human topologies found in "this" image.
        std::vector<human_t> process(feature_map_t paf, feature_map_t conf);

        /// \brief Function to process one image.
        ///
        /// \see `poseplus::paf::process(feature_map_t paf, feature_map_t conf)`.
        /// \tparam C Container<feature_map_t>
        /// \param feature_map_containers {PAF, CONF} tensors.
        /// \return All human topologies found in "this" image.
        /// \note Template parameter `C` must support `operator[]` as indexing.
        template <typename C>
        std::vector<human_t> process(C&& feature_map_containers)
        {
            // 1@paf, 2@conf.
            return process(feature_map_containers[0], feature_map_containers[1]);
        }

        ///
        /// \param thresh The PAF threshold.
        void set_paf_thresh(float thresh);

        ///
        /// \param thresh The CONF threshold.
        void set_conf_thresh(float thresh);

        /// \note This copy constructor will only copy the parameters introduces in constructor(`poseplus::paf`).
        /// \param p Object to be "copied".
        paf(const paf& p);

        /// Deconstructor.
        ~paf();

    private:
        cv::Size m_resolution_size;
        float m_paf_thresh, m_conf_thresh;

        static constexpr nullptr_t UNINITIALIZED_PTR = nullptr;
        static constexpr int UNINITIALIZED_VAL = -1;

        int m_n_joints = UNINITIALIZED_VAL, m_n_connections = UNINITIALIZED_VAL;
        cv::Size m_feature_size = { UNINITIALIZED_VAL, UNINITIALIZED_VAL };
        std::unique_ptr<ttl::tensor<float, 3>> m_upsample_conf, m_upsample_paf;
        class peak_finder_impl;
        std::unique_ptr<peak_finder_impl> m_peak_finder_ptr;
    };

} // namespace parser

} // namespace poseplus