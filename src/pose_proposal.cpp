#include "logging.hpp"
#include <deque>
#include <hyperpose/operator/parser/proposal_network.hpp>

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

    void pose_proposal::set_max_person(int n_person)
    {
        m_max_person = n_person;
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
            if (boxes.size() == 0)
                return ret;

            std::multimap<int, size_t> idxs;
            for (size_t i = 0; i < boxes.size(); ++i)
                idxs.emplace(boxes[i].second.br().y, i);

            while (idxs.size() > 0) {
                auto last = --std::end(idxs);
                const auto& box = boxes[last->second];

                idxs.erase(last);

                for (auto pos = idxs.begin(); pos != idxs.end();) {
                    const auto& box_ = boxes[pos->second];

                    float int_area = (box.second & box_.second).area();
                    float union_area = box.second.area() + box_.second.area() - int_area;
                    float overlap = int_area / union_area;

                    if (overlap > m_nms_thresh)
                        pos = idxs.erase(pos);
                    else
                        ++pos;
                }

                ret.push_back(box);
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

        // TODO. Debug
        cv::Mat debug = cv::Mat::zeros(m_net_resolution, CV_8UC(3));

        int width = debug.size().width;
        int height = debug.size().height;

        int stepSize = width / 12;

        for (int i = 0; i < height; i += stepSize)
            cv::line(debug, cv::Point(0, i), cv::Point(width, i), cv::Scalar(0, 255, 255));

        for (int i = 0; i < width; i += stepSize)
            cv::line(debug, cv::Point(i, 0), cv::Point(i, height), cv::Scalar(255, 0, 255));

        for (size_t i = 0; i < n_key_points; ++i) {
            key_point_bboxes kp_list;

            // Collect key point bounding boxes in one type.
            for (size_t j = 0; j < n_grids; ++j) {
                const size_t feature_map_index = n_grids * i + j;

                if (m_point_thresh < conf_point.view<float>()[feature_map_index])
                    kp_list.emplace_back(
                        meta_info{ (int)j, conf_point.view<float>()[feature_map_index] },
                        cv::Rect(std::max(std::min(m_net_resolution.width, static_cast<int>(x.view<float>()[feature_map_index])), 0),
                            std::max(std::min(m_net_resolution.height, static_cast<int>(y.view<float>()[feature_map_index])), 0),
                            std::max(std::min(m_net_resolution.width, static_cast<int>(w.view<float>()[feature_map_index])), 0),
                            std::max(std::min(m_net_resolution.height, static_cast<int>(h.view<float>()[feature_map_index])), 0)));
            }

            auto nms_kp_list = nms(kp_list);

            std::sort(nms_kp_list.begin(), nms_kp_list.end(), [](const std::pair<meta_info, bbox>& l, const std::pair<meta_info, bbox>& r) {
                return l.first.conf > r.first.conf;
            });

            if (nms_kp_list.size() > m_max_person)
                nms_kp_list.erase(std::next(nms_kp_list.begin(), m_max_person), nms_kp_list.end());

            info("Key Point @ ", i, " got ", nms_kp_list.size(), " proposals after nms & thresh.\n");

            key_points.push_back(std::move(nms_kp_list));
        }

        std::vector<human_t> ret_poses;

        size_t n_range = std::min(n_edges, COCOPAIR_STD.size());
        const size_t n_neighbors = h_edge_neighbor * w_edge_neighbor;
        for (size_t i = 0; i < n_range; ++i) {
            auto& from = key_points.at(COCOPAIR_STD[i].first);
            auto& to = key_points.at(COCOPAIR_STD[i].second);

            // 17 x 9 x 9 x 12 x 12
            float best_conf = m_limb_thresh;
            int best_to_id = -1;

            for (auto&& from_p : from) {
                if (!from_p.first.has_root()) {
                    from_p.first.set_root(ret_poses.size());
                    ret_poses.push_back(human_t{});
                    ret_poses.back().parts[COCOPAIR_STD[i].first] = {
                        true,
                        (float)from_p.second.x / m_net_resolution.width,
                        (float)from_p.second.y / m_net_resolution.height,
                        from_p.first.conf
                    };
                    ret_poses.back().score = 1.;
                } // from_p must have root.

                const auto& from_grid_index = from_p.first.grid_index;
                for (size_t j = 0; j < n_neighbors; ++j) {
                    const size_t edge_conf_index = i * (n_grids * n_neighbors) + j * n_grids + from_grid_index;

                    const size_t from_grid_y = from_grid_index / w_grid;
                    const size_t from_grid_x = from_grid_index - from_grid_y * w_grid;

                    const size_t aim_neighbor_y = j / w_edge_neighbor;
                    const size_t aim_neighbor_x = j - w_edge_neighbor * aim_neighbor_y;

                    const size_t aim_to_y = from_grid_y + aim_neighbor_y - h_edge_neighbor / 2;
                    const size_t aim_to_x = from_grid_x + aim_neighbor_x - w_edge_neighbor / 2;

                    bool out_of_range = (aim_to_x < 0 || aim_to_x >= w_grid || aim_to_y < 0 || aim_to_y >= h_grid);

                    if (!out_of_range && edge.view<float>()[edge_conf_index] > best_conf) {
                        for (size_t k = 0; k < to.size(); ++k) {
                            auto&& p_to = to[k];
                            size_t to_grid_y = p_to.first.grid_index / w_grid;
                            size_t to_grid_x = p_to.first.grid_index - to_grid_y * w_grid;
                            if (to_grid_x == aim_to_x && to_grid_y == aim_to_y) { // Match Point!
                                best_conf = edge.view<float>()[edge_conf_index];
                                best_to_id = k;
                                break;
                            }
                        }
                    }
                }

                if (best_to_id != -1) {
                    ret_poses[from_p.first.root()].parts[COCOPAIR_STD[i].second] = {
                        true,
                        (float)to[best_to_id].second.x / m_net_resolution.width,
                        (float)to[best_to_id].second.y / m_net_resolution.height,
                        to[best_to_id].first.conf
                    };
                    ret_poses[from_p.first.root()].score += 1.;
                }
            }
        }

        ret_poses.erase(std::remove_if(ret_poses.begin(), ret_poses.end(), [](const human_t& pose) {
            return pose.score <= 1;
        }),
            ret_poses.end());

        info("Detected ", ret_poses.size(), " human parts originally\n");

        constexpr size_t hash_grid = 64;
        constexpr size_t nan = std::numeric_limits<uint16_t>::max();
        std::array<std::array<uint16_t, hash_grid>, hash_grid> hash_table{}; // Avoid Stack OverFlow!
        for (auto&& row : hash_table)
            row.fill(nan);

        const auto query_table = [hash_grid, &hash_table](const body_part_t& part) -> uint16_t& {
            assert(part.x >= 0);
            assert(part.y >= 0);
            size_t x_ind = part.x * hash_grid;
            size_t y_ind = part.y * hash_grid;
            x_ind = (x_ind == hash_grid) ? hash_grid - 1 : x_ind;
            y_ind = (y_ind == hash_grid) ? hash_grid - 1 : y_ind;
            return hash_table[x_ind][y_ind];
        };

        for (size_t i = 0; i < ret_poses.size(); ++i) {
            auto& this_human = ret_poses[i]; // We are trying to find other parts for current human.
            for (size_t j = 0; j < this_human.parts.size(); ++j) {
                const auto& this_part = this_human.parts[j]; // Current part: Unique Or Belong to Others.
                if (this_part.has_value) {
                    const size_t root_index = query_table(this_part);
                    if (root_index >= ret_poses.size()) { // Unique.
                        query_table(this_part) = i;
                    } else {
                        if (root_index == i) // My root is the current person? Skip.
                            continue;

                        auto& aim = ret_poses[root_index]; // Get root person.

                        for (size_t u = 0; u < this_human.parts.size(); ++u) {
                            const auto& part = ret_poses[i].parts[u];
                            if (part.has_value && !aim.parts.at(u).has_value) {
                                query_table(part) = root_index;
                                aim.parts[u] = part;
                                aim.score += 1.0;
                            }
                        }

                        ret_poses.erase(ret_poses.begin() + i);
                        --i;
                        break;
                    }
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