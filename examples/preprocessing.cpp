//
// Created by ganler on 3/8/20.
//

#include "preprocessing.hpp"
#include "trace.hpp"

namespace pose
{

namespace pre
{

std::vector<internal_result_t>
tensorrt_processing::sync_push_batch(const batch_t &batch)
{

    /// Thread local static data. (For thread safety).
    thread_local ttl::tensor<uint8_t, 4> hwc_images(
        m_max_batch_size, m_inp_size.height, m_inp_size.width, 3);
    thread_local ttl::tensor<float, 4> chw_images(
        m_max_batch_size, 3, m_inp_size.height, m_inp_size.width);

    /// Return value.
    std::vector<internal_result_t> ret;
    ret.reserve(batch.size());

    {   /// Data preparation.
        TRACE_SCOPE(__func__);
        for (size_t i = 0; i < batch.size(); ++i) {
            auto hwc_buffer = hwc_images[i].data();
            auto chw_buffer = chw_images[i].data();
            cv::Mat resized_wrapper(m_inp_size, CV_8UC(3), hwc_buffer);
            cv::resize(batch[i], resized_wrapper, m_inp_size);
            ttl::tensor_ref<uint8_t, 3> s(hwc_buffer, m_inp_size.height,
                                          m_inp_size.width, 3);
            ttl::tensor_ref<float, 3> t(chw_buffer, 3, m_inp_size.height,
                                        m_inp_size.width);
            for (int k = 0; k < 3; ++k)
                for (int i = 0; i < m_inp_size.height; ++i)
                    for (int j = 0; j < m_inp_size.width; ++j)
                        t.at(k, i, j) = s.at(i, j, m_flip_rgb ? 2 - k : k) / 255.0;
        }
    }

//    {
//        TRACE_SCOPE("batch run tensorRT");
//        (*m_feature_map_solver)({chw_images.data()},
//                                {pafs.data(), confs.data()},
//                                batch.size());
//    }
}

}  // namespace pre

}  // namespace pose