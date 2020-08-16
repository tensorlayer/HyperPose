#include "../coco.hpp"
#include "../logging.hpp"
#include "fake.hpp"
#include <hyperpose/operator/parser/paf.hpp>
#include <thread>

namespace hyperpose {
namespace parser {

    struct paf::ttl_impl {
    };

    struct paf::peak_finder_impl {
    };

    paf::paf(float conf_thresh, float paf_thresh, cv::Size resolution_size)
        : m_resolution_size(resolution_size)
        , m_conf_thresh(conf_thresh)
        , m_paf_thresh(paf_thresh)
    {
        error_exit_fake();
    }

    paf::paf(const paf& p)
        : m_resolution_size(p.m_resolution_size)
        , m_conf_thresh(p.m_conf_thresh)
        , m_paf_thresh(p.m_paf_thresh)
    {
        error_exit_fake();
    }

    std::vector<human_t> paf::process(const feature_map_t& conf_map, const feature_map_t& paf_map)
    {
        std::vector<human_t> humans{};
        error_exit_fake();
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