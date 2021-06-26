#include "pifpaf_decoder/openpifpaf_postprocessor.hpp"
#include <hyperpose/operator/parser/pifpaf.hpp>

namespace hyperpose::parser {

// TODO: Name ORDER!
std::vector<human_t> pifpaf::process(const feature_map_t& paf, const feature_map_t& pif)
{
    // Helpful links (Chinese)::
    // https://zhuanlan.zhihu.com/p/93896207
    // https://zhuanlan.zhihu.com/p/68073113
    // pif: [17, 5, h, w] => KEY POINTS;
    // 5: [conf, dx, dy, b, scale]
    // Example: array([ 0.00527313,  0.13620843, -0.32253477,  0.3263721 ,  0.90980804], dtype=float32)
    // heat map: f(x, y) = \sum_ij conf * N(x, y|ij)
    // paf: [19, 9, h, w] => LIMBS;
    // 9: [conf, [x1, y1, x2, y2], [b1, b2], [s1, s2]]
    // Example: [ 0.00712654, -0.54057586,  5.4075847 ,  3.0354404 ,  3.1246614 ,  1.0621283 , -3.5857565 ,  2.6072054 ,  3.8406293 ],
    // TODO: OPTIMIZE THIS.

    lpdnn::aiapp_impl::OpenPifPafPostprocessor pp;
    pp.keypointThreshold = m_keypoint_thresh;
    size_t h = pif.shape()[pif.shape().size() - 2];
    size_t w = pif.shape().back();

    std::vector<float> pif_vec{}, paf_vec{};

    const auto raw_copy = [](const feature_map_t& tensor, std::vector<float>& vec) {
        size_t d0 = tensor.shape()[0];
        size_t d1 = tensor.shape()[1];
        size_t h = tensor.shape()[2];
        size_t w = tensor.shape()[3];
        const size_t total_size = d0 * d1 * h * w;
        vec.reserve(total_size);
        for (size_t i = 0; i < total_size; ++i) {
            vec.push_back(tensor.view<float>()[i]);
        }
    };

    raw_copy(pif, pif_vec);
    raw_copy(paf, paf_vec);

    // TODO: RECOVER THE INP{W, H};
    auto apires = pp.postprocess(m_net_w, m_net_h, w, h, pif_vec, paf_vec);

    std::vector<human_t> ret{};
    ret.reserve(apires.items.size());
    // OpenPifPaf COCO Topology: https://miro.medium.com/max/366/0*KFrFQVj3OoGAtt6o.png
    // HyperPose: Unified Topology
    // NOTE: This step is to convert pifpaf topology to hyperpose topology.

    for (auto&& item : apires.items) {
        if (item.landmarks.points.empty())
            continue;
        human_t man{};
        man.score = item.confidence;

        auto p2p = [this](const auto& src, auto& dst) {
            if (src.confidence > 0.) {
                dst.score = 1; // src.confidence; FIXME
                dst.x = src.position.x / (float)m_net_w;
                dst.y = src.position.y / (float)m_net_h;
                dst.has_value = true;
            }
        };

        auto& from = item.landmarks.points;
        auto& to = man.parts;
        // OpenPifPaf -> HyperPose
        p2p(from[0], to[0]);
        // ! to [1]
        constexpr std::array<size_t, 16> from_index = {
            6, 8, 10, 5, 7, 9,
            12, 14, 16, 11, 13, 15,
            2, 1, 4, 3
        };

        for (size_t i = 0; i < from_index.size(); ++i) {
            p2p(from[from_index[i]], to[i + 2]);
        }

        if (to[2].has_value && to[5].has_value) {
            to[1].x = (to[2].x + to[5].x) / 2;
            ;
            to[1].y = (to[2].y + to[5].y) / 2;
            ;
            to[1].has_value = true;
            to[1].score = (to[2].score + to[5].score) / 2;
        }

        ret.push_back(man);
    }

    return ret;
}

} // namespace hyperpose