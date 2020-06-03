#include <hyperpose/operator/parser/proposal_network.hpp>
#include "logging.hpp"

namespace hyperpose {

namespace parser {

std::vector<human_t> pose_proposal::process(
    const feature_map_t& conf_point, const feature_map_t& conf_iou,
    const feature_map_t& x, const feature_map_t& y, const feature_map_t& w, const feature_map_t& h,
    const feature_map_t& edge) {

    // Current Implementation Just Ignores conf_iou according to https://github.com/wangziren1/pytorch_pose_proposal_networks.

    assert(conf_point.shape().size() == 3);
    assert(std::equal(conf_point.shape().cbegin(), conf_point.shape().cend(), x.shape().cbegin()));
    assert(std::equal(conf_point.shape().cbegin(), conf_point.shape().cend(), y.shape().cbegin()));
    assert(std::equal(conf_point.shape().cbegin(), conf_point.shape().cend(), w.shape().cbegin()));
    assert(std::equal(conf_point.shape().cbegin(), conf_point.shape().cend(), h.shape().cbegin()));

    const size_t n_key_points = conf_iou.shape().front();
    const size_t w_grid = conf_point.shape()[2];
    const size_t h_grid = conf_point.shape()[1];
    const size_t n_grids = w_grid * h_grid;

    struct meta_info{
        int grid_index;
        float conf;
    };

    using bbox = cv::Rect;
    using key_point_bboxes = std::vector<std::pair<meta_info, bbox>>;

    auto nms = [this](key_point_bboxes boxes){
        key_point_bboxes ret;
        if (boxes.size() == 0)
            return ret;

        std::multimap<int, size_t> idxs;
        for (size_t i = 0; i < boxes.size(); ++i)
            idxs.emplace(boxes[i].second.br().y, i);

        while (idxs.size() > 0) {
            auto last = --std::end(idxs);
            const auto& box = boxes[last->second];

            idxs.erase(last);

            for(auto pos = idxs.begin(); pos != idxs.end(); )
            {
                const auto& box_ = boxes[pos->second];

                float int_area = (box.second & box_.second).area();
                float union_area = box.second.area() + box_.second.area() - int_area;
                float overlap = int_area / union_area;

                if (overlap > m_mns_thresh)
                    pos = idxs.erase(pos);
                else
                    ++ pos;
            }

            ret.push_back(box);
        }

        return ret;
    };

    std::vector<key_point_bboxes> key_points;
    key_points.reserve(n_key_points);

    for (size_t i = 0; i < n_key_points; ++i) {
        key_point_bboxes kp_list;

        // Collect key point bounding boxes in one type.
        for (size_t j = 0; j < n_grids; ++j) {
            size_t feature_map_index = n_grids * i + j;

            if (m_point_thresh < conf_point.view<float>()[feature_map_index])
                kp_list.emplace_back(
                    meta_info{(int)feature_map_index, conf_point.view<float>()[feature_map_index]},
                    cv::Rect(std::max(std::min(m_net_resolution.width, static_cast<int>(x.view<float>()[feature_map_index])), 0),
                             std::max(std::min(m_net_resolution.height, static_cast<int>(y.view<float>()[feature_map_index])), 0),
                             std::max(std::min(m_net_resolution.width, static_cast<int>(w.view<float>()[feature_map_index])), 0),
                             std::max(std::min(m_net_resolution.height, static_cast<int>(h.view<float>()[feature_map_index])), 0))
                );
        }

        auto nms_kp_list = nms(kp_list);


        std::sort(nms_kp_list.begin(), nms_kp_list.end(), [](const std::pair<meta_info, bbox>& l, const std::pair<meta_info, bbox>& r) {
            return l.first.conf > r.first.conf;
        });

        if (key_points.size() > m_max_person)
            key_points.erase(std::next(key_points.begin(), m_max_person), key_points.end());

        key_points.push_back(nms_kp_list);

        info("Key Point @ ", i, " got ", key_points.back().size(), " proposals after nms & thresh.\n");
    }
    return {};
}

}

}