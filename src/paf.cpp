#include "coco.hpp"
#include "logging.hpp"
#include "post_process.hpp"
#include <hyperpose/operator/parser/paf.hpp>
#include <thread>

struct connection {
    int cid1;
    int cid2;
    float score;
    int peak_id1;
    int peak_id2;
};

struct body_part_ret_t {
    int id = -1; ///< id of peak in the list of all peaks
};

template <int J>
struct human_ref_t_ {
    int id;
    body_part_ret_t parts[J];
    float score;
    int n_parts;

    human_ref_t_()
        : id(-1)
        , score(0)
        , n_parts(0)
    {
    }

    inline bool touches(const std::pair<int, int>& p, const connection& conn) const
    {
        return parts[p.first].id == conn.cid1 || parts[p.second].id == conn.cid2;
    }
};

using human_ref_t = human_ref_t_<hyperpose::COCO_N_PARTS>;

struct connection_candidate {
    int idx1;
    int idx2;
    float score;
    float etc;
};

inline bool operator>(const connection_candidate& a, const connection_candidate& b)
{
    return a.score > b.score;
}

namespace hyperpose {

namespace parser {

    constexpr int THRESH_VECTOR_CNT1 = 8;
    constexpr int THRESH_PART_CNT = 4;
    constexpr float THRESH_HUMAN_SCORE = 0.4;
    constexpr int STEP_PAF = 10;

    struct VectorXY {
        float x;
        float y;
    };

    static std::vector<VectorXY>
    get_paf_vectors(const ttl::tensor_view<float, 3>& pafmap,
        const int& ch_id1, //
        const int& ch_id2, //
        const point_2d<int>& peak1, //
        const point_2d<int>& peak2)
    {
        auto roundpaf = [](float v) { return static_cast<int>(v + 0.5); };
        std::vector<VectorXY> paf_vectors;

        const float STEP_X = (peak2.x - peak1.x) / float(STEP_PAF);
        const float STEP_Y = (peak2.y - peak1.y) / float(STEP_PAF);

        for (int i : ttl::range(STEP_PAF)) {
            int location_x = roundpaf(peak1.x + i * STEP_X);
            int location_y = roundpaf(peak1.y + i * STEP_Y);

            VectorXY v;
            v.x = pafmap.at(ch_id1, location_y, location_x);
            v.y = pafmap.at(ch_id2, location_y, location_x);
            paf_vectors.push_back(v);
        }

        return paf_vectors;
    }

    static std::vector<connection_candidate>
    get_connection_candidates(const ttl::tensor_view<float, 3>& pafmap,
        const std::vector<peak_info>& all_peaks,
        const std::vector<int>& peak_index_1,
        const std::vector<int>& peak_index_2,
        const std::pair<int, int> coco_pair_net, int height, float paf_thresh)
    {
        std::vector<connection_candidate> candidates{};

        const auto maybe_add = [&](const peak_info& peak_a, const peak_info& peak_b) {
            const auto dis = peak_b.pos - peak_a.pos;
            const float norm = std::sqrt(dis.l2());
            if (norm < 1e-12) {
                return;
            }
            point_2d<float> vec = dis.cast_to<float>();
            vec.x /= norm;
            vec.y /= norm;

            const std::vector<VectorXY> paf_vecs = get_paf_vectors(pafmap, //
                coco_pair_net.first, //
                coco_pair_net.second, //
                peak_a.pos, peak_b.pos);

            float scores = 0.0f;

            // criterion 1 : score treshold count
            int criterion1 = 0;
            for (int i = 0; i < STEP_PAF; i++) {
                const float score = vec.x * paf_vecs[i].x + vec.y * paf_vecs[i].y;
                scores += score;

                if (score > paf_thresh)
                    criterion1 += 1;
            }

            float criterion2 = scores / STEP_PAF + std::min(0.0, 0.5 * height / norm - 1.0);

            if (criterion1 > THRESH_VECTOR_CNT1 && criterion2 > 0)
                candidates.push_back(
                    { /*candidate.idx1 =*/peak_a.id,
                        /*candidate.idx2 =*/peak_b.id,
                        /*candidate.score =*/criterion2,
                        /*candidate.etc =*/criterion2 + peak_a.score + peak_b.score });
        };

        for (auto id1 : peak_index_1)
            for (auto idx2 : peak_index_2)
                maybe_add(all_peaks[id1], all_peaks[idx2]);

        return candidates;
    }

    static std::vector<human_ref_t>
    get_humans(const std::vector<peak_info>& all_peaks,
        const std::vector<std::vector<connection>>& all_connections)
    {
        TRACE_SCOPE(__func__);

        std::vector<human_ref_t> human_refs;
        for (int pair_id = 0; pair_id < COCO_N_PAIRS; pair_id++) {
            // printf("pair_id: %d, has %lu connections\n", pair_id,
            //        all_connections[pair_id].size());

            const auto coco_pair = COCOPAIRS[pair_id];
            const int part_id1 = coco_pair.first;
            const int part_id2 = coco_pair.second;

            for (const connection& conn : all_connections[pair_id]) {
                std::vector<int> hr_ids;

                for (auto hr : human_refs) {
                    if (hr.touches(coco_pair, conn)) {
                        hr_ids.push_back(hr.id);
                    }
                }
                // printf("%lu humans touches this connection\n",
                // hr_ids.size());

                if (hr_ids.size() == 1) {
                    auto& hr1 = human_refs[hr_ids[0]];
                    if (hr1.parts[part_id2].id != conn.cid2) {
                        hr1.parts[part_id2].id = conn.cid2;
                        ++hr1.n_parts;
                        hr1.score += all_peaks[conn.cid2].score + conn.score;
                    }
                } else if (hr_ids.size() >= 2) {
                    auto& hr1 = human_refs[hr_ids[0]];
                    auto& hr2 = human_refs[hr_ids[1]];

                    int membership = 0;
                    for (int i = 0; i < COCO_N_PARTS; ++i) {
                        if (hr1.parts[i].id > 0 && hr2.parts[i].id > 0) {
                            membership = 2;
                        }
                    }

                    if (membership == 0) {
                        for (int i = 0; i < COCO_N_PARTS; i++) {
                            // FIXME: double check!
                            hr1.parts[i].id += hr2.parts[i].id + 1;
                        }

                        hr1.n_parts += hr2.n_parts;
                        hr1.score += hr2.score;
                        hr1.score += conn.score;

                        human_refs.erase(human_refs.begin() + hr_ids[1]);
                    } else {
                        hr1.parts[part_id2].id = conn.cid2;
                        hr1.n_parts += 1;
                        hr1.score += all_peaks[conn.cid2].score + conn.score;
                    }
                } else if (hr_ids.size() == 0 && !is_virtual_pair(pair_id)) {
                    human_ref_t h;
                    h.parts[part_id1].id = conn.cid1;
                    h.parts[part_id2].id = conn.cid2;
                    h.n_parts = 2;
                    h.score = all_peaks[conn.cid1].score + all_peaks[conn.cid2].score + conn.score;
                    h.id = human_refs.size();

                    human_refs.push_back(h);
                }
            }
        }

        info("got ", human_refs.size(), " incomplete humans\n");

        human_refs.erase(std::remove_if(human_refs.begin(), human_refs.end(),
                             [&](const human_ref_t& hr) {
                                 return (hr.n_parts < THRESH_PART_CNT || hr.score / hr.n_parts < THRESH_HUMAN_SCORE);
                             }),
            human_refs.end());
        return human_refs;
    }

    static std::vector<connection>
    get_connections(const ttl::tensor_view<float, 3>& pafmap,
        const std::vector<peak_info>& all_peaks,
        const std::vector<std::vector<int>>& peak_ids_by_channel,
        int pair_id, int height, float paf_thresh)
    {
        const auto coco_pair = COCOPAIRS[pair_id];
        const auto coco_pair_net = COCOPAIRS_NET[pair_id];

        std::vector<connection_candidate> candidates = get_connection_candidates(
            pafmap, all_peaks, //
            peak_ids_by_channel[coco_pair.first],
            peak_ids_by_channel[coco_pair.second], coco_pair_net, height, paf_thresh);

        // nms
        std::sort(candidates.begin(), candidates.end(),
            std::greater<connection_candidate>());

        std::vector<connection> conns;
        for (const auto& candidate : candidates) {
            bool assigned = false;
            for (const auto& conn : conns) {
                if (conn.peak_id1 == candidate.idx1 || conn.peak_id2 == candidate.idx2) {
                    assigned = true;
                    break;
                }
            }
            if (!assigned) {
                connection conn;
                conn.peak_id1 = candidate.idx1;
                conn.peak_id2 = candidate.idx2;
                conn.score = candidate.score;
                conn.cid1 = candidate.idx1;
                conn.cid2 = candidate.idx2;
                conns.push_back(conn);
            }
        }
        return conns;
    }

    // Class paf.
    struct paf::peak_finder_impl : public peak_finder_t<float> {
    public:
        using peak_finder_t::peak_finder_t;
    };

    struct paf::ttl_impl {
        std::unique_ptr<ttl::tensor<float, 3>> m_upsample_paf, m_upsample_conf;
    };

    paf::paf(float conf_thresh, float paf_thresh, cv::Size resolution_size)
        : m_conf_thresh(conf_thresh)
        , m_paf_thresh(paf_thresh)
        , m_resolution_size(resolution_size)
        , m_ttl(UNINITIALIZED_PTR)
    {
    }

    paf::paf(const paf& p)
        : m_conf_thresh(p.m_conf_thresh)
        , m_paf_thresh(p.m_paf_thresh)
        , m_resolution_size(p.m_resolution_size)
        , m_ttl(UNINITIALIZED_PTR)
    {
    }

    std::vector<human_t> paf::process(const feature_map_t& conf_map, const feature_map_t& paf_map)
    {
        TRACE_SCOPE("PAF");
        std::vector<human_t> humans{};

        if (conf_map.shape().size() != 3 || paf_map.shape().size() != 3)
            error("Input of PAF::PROCESS didn't meet requirements: [conf, paf], tensor.dims() == 3\n");

        auto conf_tensor_ref = ttl::tensor_view<float, 3>(conf_map.view<float>(), conf_map.shape()[0], conf_map.shape()[1], conf_map.shape()[2]);
        auto paf_tensor_ref = ttl::tensor_view<float, 3>(paf_map.view<float>(), paf_map.shape()[0], paf_map.shape()[1], paf_map.shape()[2]);

        auto [n_connections_2_, fw_paf, fh_paf] = paf_tensor_ref.dims();
        auto [n_joints_, fw_conf, fh_conf] = conf_tensor_ref.dims();

        if (m_resolution_size.width == UNINITIALIZED_VAL || m_resolution_size.height == UNINITIALIZED_VAL)
            m_resolution_size = cv::Size(fw_paf * 4, fh_paf * 4);
        // According to OpenPose-Lightweight. It's better to be 4x feature map size.

        assert(fw_paf == fw_conf);
        assert(fh_paf == fh_conf);

        if (m_n_connections == UNINITIALIZED_VAL && m_ttl == UNINITIALIZED_PTR && m_n_joints == UNINITIALIZED_VAL) {
            m_n_connections = n_connections_2_ / 2;
            m_n_joints = n_joints_;

            m_ttl = std::make_unique<ttl_impl>();
            m_ttl->m_upsample_conf = std::make_unique<ttl::tensor<float, 3>>(n_joints_, m_resolution_size.height, m_resolution_size.width); // conf
            m_ttl->m_upsample_paf = std::make_unique<ttl::tensor<float, 3>>(n_connections_2_, m_resolution_size.height, m_resolution_size.width); // paf

            m_feature_size = cv::Size(fw_paf, fh_paf);
            m_peak_finder_ptr = std::make_unique<paf::peak_finder_impl>(
                m_n_joints, m_resolution_size.height, m_resolution_size.width, 17);
        }

        auto& m_peak_finder = *m_peak_finder_ptr;

        {
            TRACE_SCOPE("resize heatmap and PAF");
            resize_area(conf_tensor_ref, ttl::ref(*(m_ttl->m_upsample_conf)));
            resize_area(paf_tensor_ref, ttl::ref(*(m_ttl->m_upsample_paf)));
        }

        // Get all peaks.
        const auto all_peaks = m_peak_finder.find_peak_coords(
            ttl::view(*(m_ttl->m_upsample_conf)), m_conf_thresh, false /* use_gpu */);
        const auto peak_ids_by_channel = m_peak_finder.group_by(all_peaks);

        const ttl::tensor_view<float, 3>& pafmap = ttl::view(*(m_ttl->m_upsample_paf));

        std::vector<std::vector<connection>> all_connections;

        for (int pair_id = 0; pair_id < COCO_N_PAIRS; pair_id++)
            all_connections.push_back(get_connections(pafmap, all_peaks,
                peak_ids_by_channel, pair_id,
                m_feature_size.height, m_paf_thresh));

        const auto human_refs = get_humans(all_peaks, all_connections);
        info("Got ", human_refs.size(), " humans\n");

        for (const auto& hr : human_refs) {
            human_t human;
            human.score = hr.score;
            for (int i = 0; i < COCO_N_PARTS; ++i) {
                if (hr.parts[i].id != -1) {
                    human.parts[i].has_value = true;
                    const auto p = all_peaks[hr.parts[i].id];
                    human.parts[i].score = p.score;
                    human.parts[i].x = static_cast<float>(p.pos.x) / m_resolution_size.width;
                    human.parts[i].y = static_cast<float>(p.pos.y) / m_resolution_size.height;
                }
            }
            humans.push_back(human);
        }

        return humans;
    }

    void paf::set_paf_thresh(float thresh)
    {
        m_paf_thresh = thresh;
    }

    void paf::set_conf_thresh(float thresh)
    {
        m_conf_thresh = thresh;
    }

    paf::~paf() = default;

} // namespace parser

} // namespace hyperpose