#pragma once

/// \file proposal_network.hpp
/// \brief The post-processing implementation of Pose Proposal Network.

#include "../../utility/data.hpp"
#include <algorithm>
#include <numeric>
#include <utility>

namespace hyperpose {

namespace parser {

    /// \brief The post-processing implementation of Pose Proposal Network.
    /// \see https://openaccess.thecvf.com/content_ECCV_2018/papers/Sekii_Pose_Proposal_Networks_ECCV_2018_paper.pdf
    class pose_proposal {
    public:
        /// \brief Constructor of pose_proposal.
        ///
        /// \param net_resolution The input resolution of the DNN model.
        /// \param point_thresh The threshold of key points.
        /// \param limb_thresh The threshold of limbs.
        /// \param mns_thresh The threshold of NMS algorithm.
        /// \note Example of `net_resolution`: If the input resolution of your DNN model is (384 x 384), then that is the parameter.
        explicit pose_proposal(cv::Size net_resolution, float point_thresh = 0.10, float limb_thresh = 0.05, float mns_thresh = 0.3);

        /// \brief Function to infer the pose topology of given tensor.
        ///
        /// \param conf_point
        /// \param conf_iou
        /// \param x
        /// \param y
        /// \param w
        /// \param h
        /// \param edge
        ///
        /// \note To use this function, the output of your PoseProposal model should be 6 tensors: `[key point confidence, iou conf, center_x, center_y, box_width, box_height, edge confidence]`.
        /// This is natively supported by our training framework.
        ///
        /// \return A list of inferred human poses.
        std::vector<human_t> process(
            const feature_map_t& conf_point, const feature_map_t& conf_iou,
            const feature_map_t& x, const feature_map_t& y, const feature_map_t& w, const feature_map_t& h,
            const feature_map_t& edge);

        /// \brief Another form of parsing function.
        ///
        /// \param feature_map_list A list of tensors as shown in another `process` function.
        /// \return A list of inferred human poses.
        inline std::vector<human_t> process(const std::vector<feature_map_t>& feature_map_list)
        {
            assert(feature_map_list.size() == 7);
            return this->process(
                feature_map_list.at(0),
                feature_map_list.at(1),
                feature_map_list.at(2),
                feature_map_list.at(3),
                feature_map_list.at(4),
                feature_map_list.at(5),
                feature_map_list.at(6));
        }

        /// \brief Set the key point threshold.
        /// \param thresh key point threshold.
        void set_point_thresh(float thresh);

        /// \brief Set the limb threshold.
        /// \param thresh limb threshold.
        void set_limb_thresh(float thresh);

        /// \brief Set the NMS threshold.
        /// \param thresh NMS threshold.
        void set_nms_thresh(float thresh);

    private:
        cv::Size m_net_resolution;
        float m_point_thresh;
        float m_limb_thresh;
        float m_nms_thresh;
    };

}

} // namespace hyperpose