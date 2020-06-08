#include "logging.hpp"
#include <hyperpose/stream/stream.hpp>

namespace hyperpose {

basic_stream_manager::basic_stream_manager(size_t uniform_max_size, bool use_original_resolution)
    : m_use_original_resolution(use_original_resolution)
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
    m_cv_data_i.notify_one();
    //    std::cout << "Exit: " << __PRETTY_FUNCTION__ << std::endl;
}

void basic_stream_manager::read_from(cv::VideoCapture& cap)
{
    // TODO: Support camera frame counting.
    size_t supposed_decoded = cap.get(cv::CAP_PROP_FRAME_COUNT) - cap.get(cv::CAP_PROP_POS_FRAMES);
    m_remaining_num += supposed_decoded;

    size_t really_decoded = 0;
    while (cap.isOpened()) {
        cv::Mat mat;
        cap >> mat;
        if (mat.empty())
            break;
        m_input_queue.wait_until_pushed(mat);
        m_cv_data_i.notify_one();
        ++really_decoded;
    }
    size_t diff = supposed_decoded - really_decoded;
    if (diff != 0)
        m_remaining_num -= diff;
}

void basic_stream_manager::read_from(cv::Mat mat)
{
    m_input_queue.wait_until_pushed(mat);
    ++m_remaining_num;
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

        for (size_t i = 0; i < inputs.size(); ++i) {
            if (inputs[i].empty()) {
                warning("Got an empty image, skipped");
                --m_remaining_num;
            } else {
                if (!m_use_original_resolution) {
                    cv::resize(inputs[i], inputs[i], size);
                    after_resize_mats.push_back(inputs[i]);
                } else {
                    cv::Mat resized;
                    m_input_queue_replica.wait_until_pushed(inputs[i]);
                    cv::resize(inputs[i], resized, size);
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
                for (auto&& pose : poses)
                    draw_human(raw_image, pose);
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