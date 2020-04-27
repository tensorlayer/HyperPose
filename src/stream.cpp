#include <swiftpose/stream/stream.hpp>

namespace swiftpose {

basic_stream_manager::basic_stream_manager(size_t uniform_max_size)
    : m_input_queue(uniform_max_size)
    , m_input_queue_replica(uniform_max_size)
    , m_resized_queue(uniform_max_size)
    , m_after_inference_queue(uniform_max_size)
    , m_pose_sets_queue(uniform_max_size)
{
}

void basic_stream_manager::read_from(const std::vector<cv::Mat>& inputs)
{
    auto it = inputs.begin();
    size_t step_size = std::max(m_input_queue.capacity() / 2, 1ul);
    while (std::distance(it, inputs.end()) <= step_size) {
        m_input_queue.wait_until_pushed(it, std::next(it, step_size));
        std::advance(it, step_size);
    }
    m_input_queue.wait_until_pushed(it, inputs.end());
    m_remaining_num += inputs.size();
    m_cv_data_i.notify_one();
}

void basic_stream_manager::read_from(cv::VideoCapture& cap)
{
    while (cap.isOpened()) {
        cv::Mat mat;
        cap >> mat;
        m_input_queue.wait_until_pushed(std::move(mat));
        ++m_remaining_num;
        m_cv_data_i.notify_one();
    }
}

void basic_stream_manager::read_from(cv::Mat mat)
{
    m_input_queue.push(mat);
    ++m_remaining_num;
    m_cv_data_i.notify_one();
}

void basic_stream_manager::resize_from_inputs(cv::Size size)
{
    std::unique_lock lk{ m_input_queue.m_mu };
    m_cv_data_i.wait(lk, [this] { return m_input_queue.m_size > 0; });

    const int queue_size = m_input_queue.capacity();
    auto&& inputs = m_input_queue.dump_all();

    auto&& f = std::async([this, &inputs] { m_input_queue_replica.push(inputs); });

    std::vector<cv::Mat> after_resize_mats(inputs.size());
    for (size_t i = 0; i < after_resize_mats.size(); ++i)
        cv::resize(inputs[i], after_resize_mats[i], size);

    m_resized_queue.push(std::move(after_resize_mats));
    m_cv_resize.notify_one();
}

void basic_stream_manager::write_to(cv::VideoWriter& writer)
{
    std::unique_lock lk{ m_pose_sets_queue.m_mu };
    m_cv_post_processing.wait(lk,
        [this] { return m_pose_sets_queue.m_size > 0; });

    auto&& pose_set = m_pose_sets_queue.dump_all();
    for (auto&& poses : pose_set) {
        auto&& raw_image = m_input_queue_replica.dump().value();
        for (auto&& pose : poses)
            draw_human(raw_image, pose);
        writer << raw_image;
        --m_remaining_num;
    }

    m_shutdown_notifier.notify_one();
}

void basic_stream_manager::add_queue_monitor(double milli)
{
    std::lock_guard lk{ m_global_mutex };
    m_thread_tracer.push_back(std::async([this, milli] {
        while (!m_shutdown) {
            std::cout << "Reporting Queue Status:";
            std::cout << "thread_safe_queue<cv::Mat> m_input_queue -> Size = "
                      << m_input_queue.unsafe_size() << '\n'
                      << "thread_safe_queue<cv::Mat> m_input_queue_replica -> Size = "
                      << m_input_queue_replica.unsafe_size() << '\n'
                      << "thread_safe_queue<cv::Mat> m_resized_queue -> Size = "
                      << m_resized_queue.unsafe_size() << '\n'
                      << "thread_safe_queue<internal_t> m_after_inference_queue -> Size = "
                      << m_after_inference_queue.unsafe_size() << '\n'
                      << "thread_safe_queue<pose_set> m_pose_sets_queue -> Size = "
                      << m_pose_sets_queue.unsafe_size() << std::endl;
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(milli * 1ms);
        }
    }));
}

basic_stream_manager::~basic_stream_manager()
{
    std::unique_lock lk{ m_global_mutex };
    m_shutdown_notifier.wait(lk, [this] { return m_remaining_num == 0; });
    m_shutdown = true;
    m_shutdown_notifier.notify_all();
    for (auto&& x : m_thread_tracer)
        x.get();
}
}