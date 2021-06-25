#include <hyperpose/operator/parser/pifpaf.hpp>
#include "pifpaf_decoder/openpifpaf_postprocessor.hpp"

namespace hyperpose::parser {

// TODO: Name ORDER!
std::vector<human_t> pifpaf::process(const feature_map_t& paf, const feature_map_t& pif) {
    // Helpful links (Chinese):
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
    size_t h = pif.shape()[pif.shape().size() - 2];
    size_t w = pif.shape().back();
    std::vector<float> pif_conf, pif_xy, pif_s, paf_conf, paf_xy1, paf_xy2, paf_b1, paf_b2;

    const auto tensor_sharding_to_vector = [](const feature_map_t& tensor, std::vector<float>& vec, size_t dim2) {
        size_t d0 = tensor.shape()[0];
        size_t d1 = tensor.shape()[1];
        size_t h = tensor.shape()[2];
        size_t w = tensor.shape()[3];
        for (int i = 0; i < d0; ++i) {
            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < w; ++k) {
                    vec.push_back(tensor.view<float>()[
                                      i * d1 * w * h +
                                      dim2 * h * w +
                                      j * w +
                                      k
                                  ]);
                }
            }
        }
    };

    const auto tensor_sharding_to_offset_vector = [](const feature_map_t& tensor, std::vector<float>& vec, size_t dimx, size_t dimy) {
        size_t d0 = tensor.shape()[0];
        size_t d1 = tensor.shape()[1];
        size_t h = tensor.shape()[2];
        size_t w = tensor.shape()[3];
        for (int i = 0; i < d0; ++i) {
            // X first & Then Y
            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < w; ++k) {
                    vec.push_back(tensor.view<float>()[
                                      i * d1 * w * h +
                                      dimx * h * w +
                                      j * w +
                                      k
                                  ]);
                }
            }

            for (int j = 0; j < h; ++j) {
                for (int k = 0; k < w; ++k) {
                    vec.push_back(tensor.view<float>()[
                                      i * d1 * w * h +
                                      dimy * h * w +
                                      j * w +
                                      k
                                  ]);
                }
            }
        }
    };

    pif_conf.reserve(17 * h * w);
    tensor_sharding_to_vector(pif, pif_conf, 0);

    pif_xy.reserve(17 * 2 * h * w);
    tensor_sharding_to_offset_vector(pif, pif_xy, 1, 2);

    pif_s.reserve(17 * h * w);
    tensor_sharding_to_vector(pif, pif_s, 4);

    // [19, 9, h, w] -> [conf, p1, p2, b1, b2, ...]
    paf_conf.reserve(19 * h * w);
    tensor_sharding_to_vector(paf, paf_conf, 0);

    paf_xy1.reserve(2 * 19 * h * w);
    tensor_sharding_to_offset_vector(paf, paf_xy1, 1, 2);

    paf_xy2.reserve(2 * 19 * h * w);
    tensor_sharding_to_offset_vector(paf, paf_xy2, 3, 4);

    paf_b1.reserve(19 * h * w);
    tensor_sharding_to_vector(paf, paf_b1, 5);

    paf_b2.reserve(19 * h * w);
    tensor_sharding_to_vector(paf, paf_b2, 6);

    // TODO: RECOVER THE INP{W, H};
    auto apires = pp.postprocess_0_8(640, 427, w, h,
                                     pif_conf.data(), pif_xy.data(), pif_s.data(),
                                     paf_conf.data(), paf_xy1.data(), paf_xy2.data(), paf_b1.data(), paf_b2.data());

    std::vector<human_t> ret{};
    ret.reserve(apires.items.size());

    /*
     *
 OpenPifPaf COCO Topology: https://miro.medium.com/max/366/0*KFrFQVj3OoGAtt6o.png
HyperPose: Unified Topology
     *
     */

    for (auto&& item : apires.items) {
        if (item.landmarks.points.empty())
            continue;
        human_t man{};
        man.score = item.confidence;

        auto p2p = [this](const auto& src, auto& dst) {
            if (src.confidence > 0.) {
                dst.score = 1;// src.confidence; FIXME
                dst.x = src.position.x / 10000.;
                dst.y = src.position.y / 10000.;
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
            p2p(from[from_index[i]], to[i+2]);
        }

        if (to[2].has_value && to[5].has_value) {
            to[1].x = (to[2].x + to[5].x) / 2;;
            to[1].y = (to[2].y + to[5].y) / 2;;
            to[1].has_value = true;
            to[1].score = (to[2].score + to[5].score) / 2;
        }

        ret.push_back(man);
    }

    return ret;
}

} // namespace hyperpose