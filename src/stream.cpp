#include "logging.hpp"
#include <hyperpose/stream/stream.hpp>

namespace hyperpose {

basic_stream_manager::basic_stream_manager(size_t uniform_max_size, bool use_original_resolution, bool keep_ratio, cv::Size inp_size)
    : m_use_original_resolution(use_original_resolution)
    , m_keep_ratio(keep_ratio)
    , m_input_size(inp_size)
    , m_input_queue(uniform_max_size)
    , m_input_queue_replica(uniform_max_size * 4)
    , m_resized_queue(uniform_max_size)
    , m_after_inference_queue(uniform_max_size)
    , m_pose_sets_queue(uniform_max_size)
{
}

void basic_stream_manager::read_from(const std::vector<cv::Mat>& inputs)
{
    auto it = inputs.begin();
    size_t step_size = std::max(m_input_queue.capacity() / 2, size_t(1ul));
    while (std::distance(it, inputs.end()) <= step_size) {
        m_input_queue.wait_until_pushed(it, std::next(it, step_size));
        std::advance(it, step_size);
    }
    m_input_queue.wait_until_pushed(it, inputs.end());
    m_remaining_num += inputs.size();
    m_ingest += inputs.size();
    m_cv_data_i.notify_one();
}

size_t basic_stream_manager::processed_num() const noexcept
{
    return m_ingest;
}

void basic_stream_manager::read_from(cv::VideoCapture& cap)
{
    if (-1 == cap.get(cv::CAP_PROP_FRAME_COUNT))
        error("STREAM API is designed for offline video processing. Please use operator API for camera video processing.\n");

    const int supposed_decoded = std::round(cap.get(cv::CAP_PROP_FRAME_COUNT) - cap.get(cv::CAP_PROP_POS_FRAMES) + 0.5);
    m_remaining_num += supposed_decoded;

    int really_decoded = 0;
    while (cap.isOpened()) {
        cv::Mat mat;
        cap >> mat;
        ++m_ingest;
        if (mat.empty())
            break;
        m_input_queue.wait_until_pushed(mat);
        m_cv_data_i.notify_one();
        ++really_decoded;
    }
    const int diff = supposed_decoded - really_decoded;
    m_remaining_num -= diff;
}

void basic_stream_manager::read_from(cv::Mat mat)
{
    m_input_queue.wait_until_pushed(mat);
    ++m_remaining_num;
    ++m_ingest;
    m_cv_data_i.notify_one();
}

void basic_stream_manager::resize_from_inputs(cv::Size size)
{
    while (true) {
        {
            std::unique_lock lk{ m_input_queue.m_mu };
            m_cv_data_i.wait(lk, [this] { return m_input_queue.m_size > 0 || m_shutdown; });
        }

        if (m_pose_sets_queue.m_size == 0 && m_shutdown)
            break;

        auto inputs = m_input_queue.dump_all();

        std::vector<cv::Mat> after_resize_mats;
        after_resize_mats.reserve(inputs.size());

        for (auto& input : inputs) {
            if (input.empty()) {
                warning("Got an empty image, skipped");
                --m_remaining_num;
            } else {
                if (!m_use_original_resolution) {
                    if (m_keep_ratio)
                        input = non_scaling_resize(input, size);
                    else
                        cv::resize(input, input, size);
                    after_resize_mats.push_back(input);
                } else {
                    m_input_queue_replica.wait_until_pushed(input);
                    cv::Mat resized;
                    if (m_keep_ratio)
                        resized = non_scaling_resize(input, size);
                    else
                        cv::resize(input, resized, size);
                    after_resize_mats.push_back(resized);
                }
            }
        }

        if (!m_use_original_resolution)
            m_input_queue_replica.wait_until_pushed(after_resize_mats);
        m_resized_queue.wait_until_pushed(std::move(after_resize_mats));
        m_cv_resize.notify_one();
    }
}

void basic_stream_manager::write_to(cv::VideoWriter& writer)
{
    try {
        while (true) {
            {
                std::unique_lock lk{ m_pose_sets_queue.m_mu };
                m_cv_post_processing.wait(lk,
                    [this] { return m_pose_sets_queue.m_size > 0 || m_shutdown; });
            }

            if (m_pose_sets_queue.m_size == 0 && m_shutdown)
                break;

            auto pose_set = m_pose_sets_queue.dump_all();
            for (auto&& poses : pose_set) {
                auto raw_image = std::move(m_input_queue_replica.dump().value());
                for (auto&& pose : poses) {
                    if (m_keep_ratio)
                        resume_ratio(pose, raw_image.size(), m_input_size);
                    draw_human(raw_image, pose);
                }
                writer << raw_image;
                --m_remaining_num;
            }

            if (m_remaining_num == 0 && m_pose_sets_queue.m_size == 0)
                break;
        }
        m_shutdown_notifier.notify_one();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        throw;
    }
}

void basic_stream_manager::add_queue_monitor(double milli)
{
    std::lock_guard lk{ m_global_mutex };
    m_thread_tracer.push_back(std::async([this, milli] {
        while (!m_shutdown) {
            info("Reporting Stream Status:\n");
            info("Remaining frames: ", m_remaining_num, '\n');
            info("Pushed frames: ", m_ingest, '\n');
            info("Shutdown or not: ", (m_shutdown ? "SHUTDOWN" : "ALIVE"), '\n');
            info("thread_safe_queue<cv::Mat> m_input_queue -> Size = ", m_input_queue.unsafe_size(), '/', m_input_queue.capacity(), '\n');
            info("thread_safe_queue<cv::Mat> m_input_queue_replica -> Size = ", m_input_queue_replica.unsafe_size(), '/', m_input_queue_replica.capacity(), '\n');
            info("thread_safe_queue<cv::Mat> m_resized_queue -> Size = ", m_resized_queue.unsafe_size(), '/', m_resized_queue.capacity(), '\n');
            info("thread_safe_queue<internal_t> m_after_inference_queue -> Size = ", m_after_inference_queue.unsafe_size(), '/', m_after_inference_queue.capacity(), '\n');
            info("thread_safe_queue<pose_set> m_pose_sets_queue -> Size = ", m_pose_sets_queue.unsafe_size(), '/', m_pose_sets_queue.capacity(), '\n');
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(milli * 1ms);
        }
    }));
}

basic_stream_manager::~basic_stream_manager()
{
    {
        std::unique_lock lk{ m_global_mutex };
        m_shutdown_notifier.wait(lk, [this] { return m_remaining_num == 0; });
    }
    m_shutdown = true;
    m_cv_post_processing.notify_one();
    m_cv_dnn_inf.notify_one();
    m_cv_data_i.notify_one();
    m_cv_resize.notify_one();
    m_shutdown_notifier.notify_one();
    for (auto&& x : m_thread_tracer)
        x.get();
}
}