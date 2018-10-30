#include <algorithm>
#include <functional>

#include <stdtensor>

using ttl::tensor;
using ttl::tensor_ref;

#include <openpose-plus.h>

#include "post-process.h"
#include "trace.hpp"

struct VectorXY {
    float x;
    float y;
};

class paf_processor_impl : public paf_processor
{
  public:
    paf_processor_impl(int input_height, int input_width, int height, int width,
                       int n_joins /* 1 + COCO_N_PARTS */,
                       int n_connections /* COCO_N_PAIRS */,
                       int gauss_kernel_size)
        : height(height),
          width(width),
          input_height(input_height),
          input_width(input_width),
          n_joins(n_joins),
          n_connections(n_connections),
          upsample_conf(n_joins, height, width),
          upsample_paf(n_connections * 2, height, width),
          peak_finder(n_joins, height, width, gauss_kernel_size)
    {
    }

    std::vector<human_t> operator()(
        const float *conf_, /* [n_joins, input_height, input_width] */
        const float *paf_ /* [n_connections * 2, input_height, input_width] */,
        bool use_gpu)
    {
        TRACE_SCOPE("paf_processor_impl::operator()");
        {
            TRACE_SCOPE("resize heatmap and PAF");
            resize_area(tensor_ref<float, 3>((float *)conf_, n_joins,
                                             input_height, input_width),
                        upsample_conf);
            resize_area(tensor_ref<float, 3>((float *)paf_, n_connections * 2,
                                             input_height, input_width),
                        upsample_paf);
        }
        const auto all_peaks =
            peak_finder.find_peak_coords(upsample_conf, THRESH_HEAT, use_gpu);
        const auto peak_ids_by_channel = peak_finder.group_by(all_peaks);
        return process(all_peaks, peak_ids_by_channel, upsample_paf);
    }

  private:
    const float THRESH_HEAT = 0.05;
    const float THRESH_VECTOR_SCORE = 0.05;
    const int THRESH_VECTOR_CNT1 = 8;
    const int THRESH_PART_CNT = 4;
    const float THRESH_HUMAN_SCORE = 0.4;
    const int STEP_PAF = 10;

    const int height;
    const int width;
    const int input_height;
    const int input_width;
    const int n_joins;
    const int n_connections;

    tensor<float, 3> upsample_conf;  // [J, H, W]
    tensor<float, 3> upsample_paf;   // [2C, H, W]

    peak_finder_t<float> peak_finder;

    std::vector<ConnectionCandidate>
    getConnectionCandidates(const tensor<float, 3> &pafmap,
                            const std::vector<peak_info> &all_peaks,
                            const std::vector<int> &peak_index_1,
                            const std::vector<int> &peak_index_2,
                            const std::pair<int, int> coco_pair_net, int height)
    {
        std::vector<ConnectionCandidate> candidates;

        const auto maybe_add = [&](const peak_info &peak_a,
                                   const peak_info &peak_b) {
            const auto dis = peak_b.pos - peak_a.pos;
            const float norm = std::sqrt(dis.l2());
            if (norm < 1e-12) { return; }
            point_2d<float> vec = dis.cast_to<float>();
            vec.x /= norm;
            vec.y /= norm;

            const std::vector<VectorXY> paf_vecs =
                get_paf_vectors(pafmap,                //
                                coco_pair_net.first,   //
                                coco_pair_net.second,  //
                                peak_a.pos, peak_b.pos);

            float scores = 0.0f;

            // criterion 1 : score treshold count
            int criterion1 = 0;
            for (int i = 0; i < STEP_PAF; i++) {
                const float score =
                    vec.x * paf_vecs[i].x + vec.y * paf_vecs[i].y;
                scores += score;

                if (score > THRESH_VECTOR_SCORE) criterion1 += 1;
            }

            float criterion2 =
                scores / STEP_PAF + std::min(0.0, 0.5 * height / norm - 1.0);

            if (criterion1 > THRESH_VECTOR_CNT1 && criterion2 > 0) {
                ConnectionCandidate candidate;
                candidate.idx1 = peak_a.id;
                candidate.idx2 = peak_b.id;
                candidate.score = criterion2;
                candidate.etc = criterion2 + peak_a.score + peak_b.score;
                candidates.push_back(candidate);
            }
        };

        for (auto id1 : peak_index_1) {
            for (auto idx2 : peak_index_2) {
                maybe_add(all_peaks[id1], all_peaks[idx2]);
            }
        }
        return candidates;
    }

    std::vector<Connection>
    getConnections(const tensor<float, 3> &pafmap,
                   const std::vector<peak_info> &all_peaks,
                   const std::vector<std::vector<int>> &peak_ids_by_channel,
                   int pair_id, int height)
    {
        const auto coco_pair = COCOPAIRS[pair_id];
        const auto coco_pair_net = COCOPAIRS_NET[pair_id];

        std::vector<ConnectionCandidate> candidates = getConnectionCandidates(
            pafmap, all_peaks,  //
            peak_ids_by_channel[coco_pair.first],
            peak_ids_by_channel[coco_pair.second], coco_pair_net, height);

        // nms
        std::sort(candidates.begin(), candidates.end(),
                  std::greater<ConnectionCandidate>());

        std::vector<Connection> conns;
        for (const auto &candidate : candidates) {
            bool assigned = false;
            for (const auto &conn : conns) {
                if (conn.peak_id1 == candidate.idx1 ||
                    conn.peak_id2 == candidate.idx2) {
                    assigned = true;
                    break;
                }
            }
            if (!assigned) {
                Connection conn;
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

    std::vector<human_ref_t>
    getHumans(const std::vector<peak_info> &all_peaks,
              const std::vector<std::vector<Connection>> &all_connections)
    {
        TRACE_SCOPE(__func__);

        std::vector<human_ref_t> human_refs;
        for (int pair_id = 0; pair_id < COCO_N_PAIRS; pair_id++) {
            // printf("pair_id: %d, has %lu connections\n", pair_id,
            //        all_connections[pair_id].size());

            const auto coco_pair = COCOPAIRS[pair_id];
            const int part_id1 = coco_pair.first;
            const int part_id2 = coco_pair.second;

            for (const auto &conn : all_connections[pair_id]) {
                std::vector<int> hr_ids;

                for (auto hr : human_refs) {
                    if (hr.touches(coco_pair, conn)) {
                        hr_ids.push_back(hr.id);
                    }
                }
                // printf("%lu humans touches this connection\n",
                // hr_ids.size());

                if (hr_ids.size() == 1) {
                    auto &hr1 = human_refs[hr_ids[0]];
                    if (hr1.parts[part_id2].id != conn.cid2) {
                        hr1.parts[part_id2].id = conn.cid2;
                        ++hr1.n_parts;
                        hr1.score += all_peaks[conn.cid2].score + conn.score;
                    }
                } else if (hr_ids.size() >= 2) {
                    auto &hr1 = human_refs[hr_ids[0]];
                    auto &hr2 = human_refs[hr_ids[1]];

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
                    h.score = all_peaks[conn.cid1].score +
                              all_peaks[conn.cid2].score + conn.score;
                    h.id = human_refs.size();

                    human_refs.push_back(h);
                }
            }
        }

        printf("got %lu incomplete humans\n", human_refs.size());

        human_refs.erase(std::remove_if(human_refs.begin(), human_refs.end(),
                                        [&](const human_ref_t &hr) {
                                            return (hr.n_parts <
                                                        THRESH_PART_CNT ||
                                                    hr.score / hr.n_parts <
                                                        THRESH_HUMAN_SCORE);
                                        }),
                         human_refs.end());
        return human_refs;
    }

    std::vector<std::vector<Connection>>
    getAllConnections(const tensor<float, 3> &pafmap,
                      const std::vector<peak_info> &all_peaks,
                      const std::vector<std::vector<int>> &peak_ids_by_channel)
    {
        TRACE_SCOPE(__func__);

        std::vector<std::vector<Connection>> all_connections;
        for (int pair_id = 0; pair_id < COCO_N_PAIRS; pair_id++) {
            all_connections.push_back(getConnections(
                pafmap, all_peaks, peak_ids_by_channel, pair_id, height));
        }
        return all_connections;
    }

    std::vector<human_t>
    process(const std::vector<peak_info> &all_peaks,
            const std::vector<std::vector<int>> &peak_ids_by_channel,
            const tensor<float, 3> &pafmap /* [2c, h, w] */)
    {
        TRACE_SCOPE("paf_processor_impl::process");

        const std::vector<std::vector<Connection>> all_connections =
            getAllConnections(pafmap, all_peaks, peak_ids_by_channel);

        const auto human_refs = getHumans(all_peaks, all_connections);
        printf("got %lu humans\n", human_refs.size());

        {
            TRACE_SCOPE("generate output");
            std::vector<human_t> humans;
            for (const auto &hr : human_refs) {
                human_t human;
                human.score = hr.score;
                for (int i = 0; i < COCO_N_PARTS; ++i) {
                    if (hr.parts[i].id != -1) {
                        human.parts[i].has_value = true;
                        const auto p = all_peaks[hr.parts[i].id];
                        human.parts[i].score = p.score;
                        human.parts[i].x = p.pos.x;
                        human.parts[i].y = p.pos.y;
                    }
                }
                humans.push_back(human);
            }
            return humans;
        }
    }

    std::vector<VectorXY> get_paf_vectors(const tensor<float, 3> &pafmap,
                                          const int &ch_id1,           //
                                          const int &ch_id2,           //
                                          const point_2d<int> &peak1,  //
                                          const point_2d<int> &peak2)
    {
        std::vector<VectorXY> paf_vectors;

        const float STEP_X = (peak2.x - peak1.x) / float(STEP_PAF);
        const float STEP_Y = (peak2.y - peak1.y) / float(STEP_PAF);

        for (int i = 0; i < STEP_PAF; i++) {
            int location_x = roundpaf(peak1.x + i * STEP_X);
            int location_y = roundpaf(peak1.y + i * STEP_Y);

            VectorXY v;
            v.x = pafmap.at(ch_id1, location_y, location_x);
            v.y = pafmap.at(ch_id2, location_y, location_x);
            paf_vectors.push_back(v);
        }

        return paf_vectors;
    }

    int roundpaf(float v) { return (int)(v + 0.5); }
};

paf_processor *create_paf_processor(int input_height, int input_width,
                                    int height, int width, int n_joins,
                                    int n_connections, int gauss_kernel_size)
{
    return new paf_processor_impl(input_height, input_width, height, width,
                                  n_joins, n_connections, gauss_kernel_size);
}
