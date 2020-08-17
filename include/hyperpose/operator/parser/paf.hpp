#pragma once

/// \file paf.hpp
/// \brief Post-processing using Part Affinity Field (PAF).

#include "../../utility/data.hpp"

namespace hyperpose {

/// \brief The namespace to contain things related to post processing.
/// \note In HyperPose, the pose estimation pipeline consists of DNN inference and parsing(post-processing). The
/// parser part implementation is under the namespace `hyperpose::parser`.
namespace parser {

    /// \brief Post-processing using Part Affinity Field (PAF).
    /// \see https://arxiv.org/abs/1812.08008
    class paf {
    public:
        /// \brief Constructor indicating the image size and thresholds.
        ///
        /// \param conf_thresh The activation threshold.
        /// \param paf_thresh The threshold of Part Affinity Field.
        /// \param resolution_size The size(width, height) of expected resolution for the post-processing.
        /// \note Before doing PAF, the (width, height) of feature map will be expanded to `resolution_size` to perform
        /// a more accurate post processing. And `resolution_size` will be N x the size of first input tensor if it's
        /// not set. (now, N is 4)
        explicit paf(float conf_thresh = 0.05, float paf_thresh = 0.05, cv::Size resolution_size = cv::Size(UNINITIALIZED_VAL, UNINITIALIZED_VAL));

        /// \brief Function to process one image.
        ///
        /// \code
        /// // Initialization of PAF.
        /// hyperpose::parser::paf paf_processor();
        ///
        /// // ...
        ///
        /// // Do inference.
        /// // Note that, tensor_pairs.size is equal to the batch_size process. Each of them represents one image.
        /// auto tensor_pairs = engine.inference(...); // Ordered by tensor name.
        ///
        /// for(auto&& tensor_pair/* CONF, PAF */ : tensor_pairs)
        ///    human_topology = paf_processor.process(tensor_pair[0], tensor_pair[1]);
        /// \endcode
        ///
        /// \param conf The conf tensor.
        /// \param paf The paf tensor.
        /// \return All human topologies found in "this" image.
        std::vector<human_t> process(const feature_map_t& conf, const feature_map_t& paf);

        /// \brief Function to process one image.
        ///
        /// \see `hyperpose::paf::process(feature_map_t paf, feature_map_t conf)`.
        /// \tparam C Container<feature_map_t>
        /// \param feature_map_containers {CONF, PAF} tensors.
        /// \return All human topologies found in "this" image.
        /// \note Template parameter `C` must support `operator[]` as indexing.
        template <typename C>
        std::vector<human_t> process(C&& feature_map_containers)
        {
            // 1@conf, 2@paf.
            return process(feature_map_containers[0], feature_map_containers[1]);
        }

        ///
        /// \param thresh The PAF threshold.
        void set_paf_thresh(float thresh);

        ///
        /// \param thresh The CONF threshold.
        void set_conf_thresh(float thresh);

        /// \note This copy constructor will only copy the parameters introduces in constructor(`hyperpose::paf`).
        /// \param p Object to be "copied".
        paf(const paf& p);

        /// Deconstructor.
        ~paf();

    private:
        static constexpr std::nullptr_t UNINITIALIZED_PTR = nullptr;
        static constexpr int UNINITIALIZED_VAL = -1;

        float m_conf_thresh, m_paf_thresh;
        cv::Size m_resolution_size;
        int m_n_joints = UNINITIALIZED_VAL, m_n_connections = UNINITIALIZED_VAL;
        cv::Size m_feature_size = { UNINITIALIZED_VAL, UNINITIALIZED_VAL };

        struct ttl_impl;
        std::unique_ptr<ttl_impl> m_ttl;

        struct peak_finder_impl;
        std::unique_ptr<peak_finder_impl> m_peak_finder_ptr;
    };

} // namespace parser

} // namespace hyperpose