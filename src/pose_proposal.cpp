#include "logging.hpp"
#include <deque>
#include <hyperpose/operator/parser/proposal_network.hpp>

#include "coco.hpp"
#include "color.hpp"

namespace hyperpose {

namespace parser {

    // The pose_proposal supports the feature map list using the following order:

    // 0: conf_point         N x 18 x 12 x 12
    // 1: conf_iou           N x 18 x 12 x 12 :: Let's ignore this.
    // 2: x                  N x 18 x 12 x 12
    // 3: y                  N x 18 x 12 x 12
    // 4: w                  N x 18 x 12 x 12
    // 5: h                  N x 18 x 12 x 12
    // 6: edge_confidence    N x 17 x 9 x 9 x 12 x 12

    // -> Return human_t {x, y} \in [0, 1]]
    inline const coco_pair_list_t COCOPAIR_STD = {
        { 1, 8 }, // 0
        { 8, 9 }, // 1
        { 9, 10 }, // 2
        { 1, 11 }, // 3
        { 11, 12 }, // 4
        { 12, 13 }, // 5
        { 1, 2 }, // 6
        { 2, 3 }, // 7
        { 3, 4 }, // 8
        { 1, 5 }, // 10
        { 5, 6 }, // 11
        { 6, 7 }, // 12
        { 1, 0 }, // 14
        { 0, 14 }, // 15
        { 0, 15 }, // 16
        { 14, 16 }, // 17
        { 15, 17 }, // 18
    }; // See https://www.cnblogs.com/caffeaoto/p/7793994.html.

    pose_proposal::pose_proposal(cv::Size net_resolution, float point_thresh, float limb_thresh, float mns_thresh)
        : m_net_resolution(std::move(net_resolution))
        , m_point_thresh(point_thresh)
        , m_limb_thresh(limb_thresh)
        , m_nms_thresh(mns_thresh)
    {
    }

    void pose_proposal::set_point_thresh(float thresh)
    {
        m_point_thresh = thresh;
    }

    void pose_proposal::set_limb_thresh(float thresh)
    {
        m_limb_thresh = thresh;
    }

    void pose_proposal::set_nms_thresh(float thresh)
    {
        m_nms_thresh = thresh;
    }

    constexpr int MIN_REQUIRED_POINTS_FOR_A_MAN = 3; // 3 Connection to be a man;

    std::vector<human_t> pose_proposal::process(
        const feature_map_t& conf_point, const feature_map_t& conf_iou,
        const feature_map_t& x, const feature_map_t& y, const feature_map_t& w, const feature_map_t& h,
        const feature_map_t& edge)
    {

        // Current Implementation Just Ignores conf_iou according to https://github.com/wangziren1/pytorch_pose_proposal_networks.

        assert(conf_point.shape().size() == 3);
        assert(std::equal(conf_point.shape().cbegin(), conf_point.shape().cend(), x.shape().cbegin()));
        assert(std::equal(conf_point.shape().cbegin(), conf_point.shape().cend(), y.shape().cbegin()));
        assert(std::equal(conf_point.shape().cbegin(), conf_point.shape().cend(), w.shape().cbegin()));
        assert(std::equal(conf_point.shape().cbegin(), conf_point.shape().cend(), h.shape().cbegin()));

        const size_t n_key_points = conf_iou.shape().front();
        const size_t w_grid = conf_point.shape()[2];
        const size_t h_grid = conf_point.shape()[1];
        const size_t w_edge_neighbor = edge.shape()[2];
        const size_t h_edge_neighbor = edge.shape()[1];
        const size_t n_edges = edge.shape()[0];
        const size_t n_grids = w_grid * h_grid;

        struct meta_info {
            int grid_index;
            float conf;
            bool has_root() { return human_index != -1; }
            void set_root(int r) { human_index = r; }
            const int& root() { return human_index; }
            meta_info(int gi, float c)
                : grid_index(gi)
                , conf(c)
                , human_index(-1)
            {
            }

        private:
            int human_index = -1;
        };

        using bbox = cv::Rect;
        using key_point_bboxes = std::vector<std::pair<meta_info, bbox>>;

        auto nms = [this](key_point_bboxes boxes) {
            key_point_bboxes ret;

            std::sort(boxes.begin(), boxes.end(), [](const std::pair<meta_info, bbox>& l, const std::pair<meta_info, bbox>& r) {
                return l.first.conf < r.first.conf;
            });

            const auto iou = [](const auto& l, const auto& r) {
                float int_area = (l.second & r.second).area();
                float union_area = l.second.area() + r.second.area() - int_area;
                float overlap = int_area / union_area;
                return overlap;
            };

            while (!boxes.empty()) {
                ret.emplace_back(boxes.back());
                boxes.pop_back();
                for (size_t i = 0; i < boxes.size(); i++)
                    if (iou(ret.back(), boxes[i]) >= m_nms_thresh)
                        boxes.erase(boxes.begin() + i);
            }

            return ret;
        };

        struct human_point {
            int x, y;
            int key_point_type;
            int human_index = -1;
        };

        std::vector<key_point_bboxes> key_points;
        key_points.reserve(n_key_points);

        for (size_t i = 0; i < n_key_points; ++i) {
            key_point_bboxes kp_list;

            // Collect key point bounding boxes in one type.
            for (size_t j = 0; j < n_grids; ++j) {
                const size_t feature_map_index = n_grids * i + j;

                if (m_point_thresh < conf_point.view<float>()[feature_map_index])
                    kp_list.emplace_back(
                        meta_info{ (int)j, conf_point.view<float>()[feature_map_index] },
                        cv::Rect(std::max(std::min(m_net_resolution.width, static_cast<int>(x.view<float>()[feature_map_index] - w.view<float>()[feature_map_index] / 2)), 0),
                            std::max(std::min(m_net_resolution.height, static_cast<int>(y.view<float>()[feature_map_index] - h.view<float>()[feature_map_index] / 2)), 0),
                            std::max(std::min(m_net_resolution.width, static_cast<int>(w.view<float>()[feature_map_index])), 0),
                            std::max(std::min(m_net_resolution.height, static_cast<int>(h.view<float>()[feature_map_index])), 0)));
            }

            auto nms_kp_list = nms(std::move(kp_list));

            info("Key Point @ ", i, " got ", nms_kp_list.size(), " bounding boxes after thresh + NMS.\n");

            key_points.push_back(std::move(nms_kp_list));
        }

        std::vector<human_t> ret_poses;

        size_t n_range = std::min(n_edges, COCOPAIR_STD.size());
        const size_t n_neighbors = h_edge_neighbor * w_edge_neighbor;

        for (size_t i = 0; i < n_range; ++i) {
            auto& from = key_points.at(COCOPAIR_STD[i].first);
            auto& to = key_points.at(COCOPAIR_STD[i].second);

            struct limb {
                int from, to;
                float conf;
            };

            std::vector<limb> limb_candidates{};

            // 17 x 9 x 9 x 12 x 12
            for (size_t from_index = 0; from_index < from.size(); ++from_index) {
                auto& from_p = from[from_index];
                const auto& from_grid_index = from_p.first.grid_index; // Location of start point in the feature map.
                for (size_t j = 0; j < n_neighbors; ++j) {
                    const size_t edge_conf_index = i * (n_grids * n_neighbors) + j * n_grids + from_grid_index;

                    const size_t from_grid_y = from_grid_index / w_grid;
                    const size_t from_grid_x = from_grid_index - from_grid_y * w_grid;

                    const size_t aim_neighbor_y = j / w_edge_neighbor;
                    const size_t aim_neighbor_x = j - w_edge_neighbor * aim_neighbor_y;

                    const size_t aim_to_y = from_grid_y + aim_neighbor_y - h_edge_neighbor / 2;
                    const size_t aim_to_x = from_grid_x + aim_neighbor_x - w_edge_neighbor / 2;

                    bool out_of_range = (aim_to_x < 0 || aim_to_x >= w_grid || aim_to_y < 0 || aim_to_y >= h_grid);
                    auto possible_connection_conf = edge.view<float>()[edge_conf_index];
                    if (!out_of_range && possible_connection_conf > m_limb_thresh) {
                        for (size_t to_index = 0; to_index < to.size(); ++to_index) {
                            auto&& p_to = to[to_index];
                            size_t to_grid_y = p_to.first.grid_index / w_grid;
                            size_t to_grid_x = p_to.first.grid_index - to_grid_y * w_grid;
                            if (to_grid_x == aim_to_x && to_grid_y == aim_to_y) { // Match Point!
                                limb_candidates.push_back({ (int)from_index, (int)to_index, possible_connection_conf });
                            }
                        }
                    }
                }
                //            if (best_to_id != -1) {
                //                ret_poses[from_p.first.root()].parts[COCOPAIR_STD[i].second] = {
                //                    true,
                //                    (float)(to[best_to_id].second.x + to[best_to_id].second.width / 2) / m_net_resolution.width,
                //                    (float)(to[best_to_id].second.y + to[best_to_id].second.height / 2) / m_net_resolution.height,
                //                    to[best_to_id].first.conf
                //                };
                //                ret_poses[from_p.first.root()].score += 1.;
                //            }
            }

            // All right. We now get all possible [from, to] pairs. Let's choose them by rank.
            std::sort(limb_candidates.begin(), limb_candidates.end(), [](auto& l, auto& r) {
                return l.conf < r.conf;
            });

            std::vector<bool> from_check(from.size(), false);
            std::vector<bool> to_check(to.size(), false);
            while (!limb_candidates.empty()) {
                auto cur = limb_candidates.back();
                limb_candidates.pop_back();

                if (from_check[cur.from] || to_check[cur.to]) // Point already taken.
                    continue;

                auto& from_val = from[cur.from];
                auto& to_val = to[cur.to];

                size_t root_index = [&]() -> size_t {
                    if (from_val.first.has_root() == to_val.first.has_root()) {
                        ret_poses.emplace_back();
                        return ret_poses.size() - 1;
                    }

                    return from_val.first.has_root() ? from_val.first.root() : to_val.first.root();
                }();

                if (!ret_poses[root_index].parts[COCOPAIR_STD[i].first].has_value) {
                    ret_poses[root_index].parts[COCOPAIR_STD[i].first] = {
                        true,
                        (float)(from_val.second.x + from_val.second.width / 2) / m_net_resolution.width,
                        (float)(from_val.second.y + from_val.second.height / 2) / m_net_resolution.height,
                        from_val.first.conf
                    };
                    from_val.first.set_root(root_index);
                    ret_poses[root_index].score += 1.;
                }

                if (!ret_poses[root_index].parts[COCOPAIR_STD[i].second].has_value) {
                    ret_poses[root_index].parts[COCOPAIR_STD[i].second] = {
                        true,
                        (float)(to_val.second.x + to_val.second.width / 2) / m_net_resolution.width,
                        (float)(to_val.second.y + to_val.second.height / 2) / m_net_resolution.height,
                        to_val.first.conf
                    };
                    to_val.first.set_root(root_index);
                    ret_poses[root_index].score += 1.;
                }
            }
        }

        info("Detected ", ret_poses.size(), " human parts originally\n");

        constexpr size_t grid_size = 64;
        std::array<std::array<std::vector<uint16_t>, grid_size>, grid_size> hash_table{};

        const auto query_table = [grid_size, &hash_table](const body_part_t& part) -> std::vector<uint16_t>& {
            assert(part.x >= 0);
            assert(part.y >= 0);
            size_t x_ind = part.x * grid_size;
            size_t y_ind = part.y * grid_size;
            x_ind = (x_ind == grid_size) ? grid_size - 1 : x_ind;
            y_ind = (y_ind == grid_size) ? grid_size - 1 : y_ind;
            return hash_table[x_ind][y_ind];
        };

        for (size_t i = 0; i < ret_poses.size(); ++i) {
            auto& cur_human = ret_poses[i]; // We are trying to find other parts for current human.
            if (cur_human.score > n_key_points - 0.1)
                continue;

            for (size_t j = 0; j < cur_human.parts.size(); ++j) {
                const auto& this_part = cur_human.parts[j]; // Current part: Unique Or Belong to Others.
                if (this_part.has_value) {
                    auto& maybes = query_table(this_part);

                    bool remove_cur = false;
                    for (auto possible_id : maybes) {
                        auto& maybe_combine = ret_poses[possible_id];
                        if (possible_id == i || maybe_combine.parts[j].y != this_part.y || maybe_combine.parts[j].x != this_part.x)
                            continue;

                        remove_cur = true;
                        for (size_t u = 0; u < cur_human.parts.size(); ++u) {
                            const auto& cur_part = cur_human.parts[u];
                            if (cur_part.has_value && !maybe_combine.parts[u].has_value) {
                                maybe_combine.parts[u] = cur_part;
                                maybe_combine.score += 1.0;
                                query_table(cur_part).push_back(i);
                            }
                        }

                        ret_poses.erase(ret_poses.begin() + i);
                        --i;
                        break;
                    }

                    if (remove_cur)
                        break;

                    maybes.push_back(i);
                }
            }
        }

        info("Combined to ", ret_poses.size(), " human parts.\n");

        ret_poses.erase(std::remove_if(ret_poses.begin(), ret_poses.end(), [](const human_t& pose) {
            return pose.score <= MIN_REQUIRED_POINTS_FOR_A_MAN;
        }),
            ret_poses.end());

        info("Got to ", ret_poses.size(), " humans finally.\n");

        return ret_poses;
    }

}

}