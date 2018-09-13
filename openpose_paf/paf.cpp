#include "paf.h"

#include <algorithm>
#include <cmath>
#include <functional>

#include "coco.h"
#include "human.h"
#include "post-process.h"
#include "tensor.h"
#include "tracer.h"
#include "vis.h"

const float THRESH_VECTOR_SCORE = 0.05;
const int THRESH_VECTOR_CNT1 = 8;
const int THRESH_PART_CNT = 4;
const float THRESH_HUMAN_SCORE = 0.4;

const int STEP_PAF = 10;

template <typename T> T sqr(T x) { return x * x; }

template <typename T> struct point_2d {
    T x;
    T y;

    point_2d<T> operator-(const point_2d<T> &p) const
    {
        return point_2d<T>{x - p.x, y - p.y};
    }

    template <typename S> point_2d<S> cast_to() const
    {
        return point_2d<S>{S(x), S(y)};
    }

    T l2() const { return sqr(x) + sqr(y); }
};

struct Peak {
    point_2d<int> pos;
    float score;
    int id;
};

struct VectorXY {
    float x;
    float y;
};

struct paf_processor {
    const float THRESH_HEAT = 0.05;

    const int height;
    const int width;
    const int n_joins;        // number of joins == 18 + 1 (background)
    const int n_connections;  // number of connections == 17 + 2 (virtual)

    paf_processor(int height, int width, int n_joins, int n_connections)
        : height(height), width(width), n_joins(n_joins),
          n_connections(n_connections)
    {
    }

    std::vector<ConnectionCandidate>
    getConnectionCandidates(const tensor_t<float, 3> &pafmap,
                            const std::vector<Peak> &all_peaks,
                            const std::vector<int> &peak_index_1,
                            const std::vector<int> &peak_index_2,
                            const std::pair<int, int> coco_pair_net, int height)
    {
        std::vector<ConnectionCandidate> candidates;

        const auto maybe_add = [&](const Peak &peak_a, const Peak &peak_b) {
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
    getConnections(const tensor_t<float, 3> &pafmap,
                   const std::vector<Peak> &all_peaks,
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

    void
    select_peaks(const tensor_t<float, 3> &heatmap,
                 const tensor_t<float, 3> &peaks,
                 float threshold,                                    //
                 std::vector<Peak> &all_peaks,                       // output
                 std::vector<std::vector<int>> &peak_ids_by_channel  // output
    )
    {
        TRACE(__func__);
        int idx = 0;
        for (int part_id = 0;
             part_id < n_joins - /* the last one is background */ 1;
             ++part_id) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (peaks.at(part_id, y, x) > threshold) {
                        const Peak info =
                            Peak{point_2d<int>{x, y}, heatmap.at(part_id, y, x),
                                 idx++};
                        all_peaks.push_back(info);
                        peak_ids_by_channel[part_id].push_back(info.id);
                    }
                }
            }
        }
    }

    std::vector<human_ref_t>
    getHumans(const std::vector<Peak> &all_peaks,
              const std::vector<std::vector<Connection>> &all_connections)
    {
        TRACE(__func__);

        std::vector<human_ref_t> human_refs;
        for (int pair_id = 0; pair_id < n_connections; pair_id++) {
            printf("pair_id: %d, has %lu connections\n", pair_id,
                   all_connections[pair_id].size());

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
                printf("%lu humans touches this connection\n", hr_ids.size());

                if (hr_ids.size() == 1) {
                    const int humans_idx1 = hr_ids[0];

                    if (human_refs[humans_idx1].parts[part_id2].id !=
                        conn.cid2) {
                        human_refs[humans_idx1].parts[part_id2].id = conn.cid2;
                        ++human_refs[humans_idx1].n_parts;
                        human_refs[humans_idx1].score +=
                            all_peaks[conn.cid2].score + conn.score;
                    }
                } else if (hr_ids.size() >= 2) {
                    const int humans_idx1 = hr_ids[0];
                    const int humans_idx2 = hr_ids[1];

                    int membership = 0;
                    for (int i = 0; i < 18; ++i) {
                        if (human_refs[humans_idx1].parts[i].id > 0 &&
                            human_refs[humans_idx2].parts[i].id > 0) {
                            membership = 2;
                        }
                    }

                    if (membership == 0) {
                        for (int humans_id = 0; humans_id < 18; humans_id++)
                            human_refs[humans_idx1].parts[humans_id].id +=
                                (human_refs[humans_idx2].parts[humans_id].id +
                                 1);

                        human_refs[humans_idx1].n_parts +=
                            human_refs[humans_idx2].n_parts;
                        human_refs[humans_idx1].score +=
                            human_refs[humans_idx2].score;
                        human_refs[humans_idx1].score += conn.score;

                        human_refs.erase(human_refs.begin() + humans_idx2);
                    } else {
                        human_refs[humans_idx1].parts[part_id2].id = conn.cid2;
                        human_refs[humans_idx1].n_parts += 1;
                        human_refs[humans_idx1].score +=
                            all_peaks[conn.cid2].score + conn.score;
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
                                        [](const human_ref_t &hr) {
                                            return (hr.n_parts <
                                                        THRESH_PART_CNT ||
                                                    hr.score / hr.n_parts <
                                                        THRESH_HUMAN_SCORE);
                                        }),
                         human_refs.end());
        return human_refs;
    }

    std::vector<std::vector<Connection>>
    getAllConnections(const tensor_t<float, 3> &pafmap,
                      const std::vector<Peak> &all_peaks,
                      const std::vector<std::vector<int>> &peak_ids_by_channel)
    {
        TRACE(__func__);

        std::vector<std::vector<Connection>> all_connections;
        for (int pair_id = 0; pair_id < n_connections; pair_id++) {
            all_connections.push_back(getConnections(
                pafmap, all_peaks, peak_ids_by_channel, pair_id, height));
        }
        return all_connections;
    }

    std::vector<human_t>
    operator()(const tensor_t<float, 3> &heatmap,  // [j, h, w]
               const tensor_t<float, 3> &peaks,    // [j, h, w]
               const tensor_t<float, 3> &pafmap    // [2c, h, w]
    )
    {
        TRACE(__func__);

        std::vector<Peak> all_peaks;
        std::vector<std::vector<int>> peak_ids_by_channel(n_joins);
        select_peaks(heatmap, peaks, THRESH_HEAT, all_peaks,
                     peak_ids_by_channel);

        const std::vector<std::vector<Connection>> all_connections =
            getAllConnections(pafmap, all_peaks, peak_ids_by_channel);

        const auto human_refs = getHumans(all_peaks, all_connections);
        printf("got %lu humans\n", human_refs.size());

        std::vector<human_t> humans;
        for (const auto &hr : human_refs) {
            human_t human;
            human.score = hr.score;
            for (int i = 0; i < 18; ++i) {
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

    std::vector<VectorXY> get_paf_vectors(const tensor_t<float, 3> &pafmap,
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

void process_conf_paf(int height_, int width_,  //
                      int channel_j,            // channel_j = n_joins
                      int channel_c,            // channel_c = n_connections
                      const float *conf_,       // [height, width, channel_j]
                      const float *paf_  // [height, width, channel_c * 2]
)
{
    TRACE(__func__);

    const int height = height_ * 8;
    const int width = width_ * 8;

    tensor_t<float, 3> upsample_conf(nullptr, channel_j, height, width);
    {
        tensor_t<float, 3> conf(nullptr, channel_j, height_, width_);
        chw_from_hwc(conf, conf_);
        resize_area(conf, upsample_conf);
    }
    tensor_t<float, 3> peaks(nullptr, channel_j, height, width);
    get_peak_map(upsample_conf, peaks);

    tensor_t<float, 3> upsample_paf(nullptr, channel_c * 2, height, width);
    {
        tensor_t<float, 3> paf(nullptr, channel_c * 2, height_, width_);
        chw_from_hwc(paf, paf_);
        resize_area(paf, upsample_paf);
    }

    paf_processor p(height, width, channel_j, channel_c);
    const auto humans = p(upsample_conf, peaks, upsample_paf);

    cv::Mat img(cv::Size(width, height),
                CV_8UC(3));  // cv::DataType<cv::Vec3b>::type);
    for (const auto h : humans) {
        h.print();
        draw_human(img, h);
    }
    cv::imwrite("result.png", img);
}
