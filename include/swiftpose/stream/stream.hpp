#pragma once

#include <future>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../utility/thread_pool.hpp"
#include "../utility/thread_safe_queue.hpp"

namespace swiftpose {

class basic_stream_manager {
public:
    basic_stream_manager(size_t uniform_max_size)
        : m_input_queue(uniform_max_size)
        , m_input_queue_replica(uniform_max_size)
        , m_resized_queue(uniform_max_size)
        , m_after_inference_queue(uniform_max_size)
        , m_pose_sets_queue(uniform_max_size)
    {
    }

    void read_from_opencv_mat(const std::vector<cv::Mat>& inputs)
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

    void read_from_opencv_video(cv::VideoCapture& cap)
    {
        while (cap.isOpened()) {
            cv::Mat mat;
            cap >> mat;
            m_input_queue.wait_until_pushed(std::move(mat));
            ++m_remaining_num;
            m_cv_data_i.notify_one();
        }
    }

    void read_from_opencv_mat(cv::Mat mat)
    {
        m_input_queue.push(mat);
        ++m_remaining_num;
        m_cv_data_i.notify_one();
    }

    void resize_from_inputs(cv::Size size)
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

    template <typename Engine>
    void dnn_inference_from_resized_images(Engine&& engine)
    {
        std::unique_lock lk{ m_resized_queue.m_mu };
        m_cv_resize.wait(lk, [this] { return m_resized_queue.m_size > 0; });

        auto&& resized_inputs = m_resized_queue.dump(engine.max_batch_size());
        auto&& internals = engine.inference(std::move(resized_inputs));
        m_after_inference_queue.push(std::move(internals));

        m_cv_dnn_inf.notify_one();
    }

    template <typename ParserList>
    void parse_from_internals(ParserList&& parser_list)
    {
        std::unique_lock lk{ m_after_inference_queue.m_mu };
        m_cv_dnn_inf.wait(lk,
            [this] { return m_after_inference_queue.m_size > 0; });

        auto&& internals = m_after_inference_queue.dump_all();

        std::vector<std::future<pose_set>> futures;
        futures.reserve(internals.size());

        for (size_t round_robin = 0; round_robin < internals.size();
             ++round_robin) {
            futures.push_back(m_thread_pool.enqueue(
                [&](size_t robin) {
                    return parser_list.at(robin % parser_list.size())
                        .process(std::move(internals[round_robin]));
                },
                round_robin));
            if ((1 + round_robin) % parser_list.size() == 0 && round_robin != 1)
                futures[round_robin + 1 - parser_list.size()].wait();
        }

        std::vector<pose_set> pose_sets;
        pose_sets.reserve(internals.size());

        for (auto&& f : futures)
            pose_sets.push_back(f.get());

        m_pose_sets_queue.push(std::move(pose_sets));
        m_cv_post_processing.notify_one();
    };

    void write_to_opencv_video(cv::VideoWriter& writer)
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

    template <typename NameGetter>
    void write_to_opencv_images(NameGetter&& name_getter)
    {
        static_assert(std::is_same_v<std::result_of_t<NameGetter()>, std::string>,
            "The return type of NameGetter should be std::string");

        std::unique_lock lk{ m_pose_sets_queue.m_mu };
        m_cv_post_processing.wait(lk,
            [this] { return m_pose_sets_queue.m_size > 0; });

        auto&& pose_set = m_pose_sets_queue.dump_all();
        for (auto&& poses : pose_set) {
            auto&& raw_image = m_input_queue_replica.dump().value();
            for (auto&& pose : poses)
                draw_human(raw_image, pose);
            cv::imwrite(name_getter(), raw_image);
            --m_remaining_num;
        }

        m_shutdown_notifier.notify_one();
    }

    void add_queue_monitor(double milli = 1000)
    {
        std::lock_guard lk{ m_global_mutex };
        m_thread_tracer.push_back(std::async([this, milli] {
            while (!m_shutdown) {
                std::cout << "Reporting Queue Status:";
                std::cout << "thread_safe_queue<cv::Mat> m_input_queue -> Size = "
                          << m_input_queue.unsafe_size() << '\n'
                          << "thread_safe_queue<cv::Mat> m_input_queue_replica -> "
                             "Size = "
                          << m_input_queue_replica.unsafe_size() << '\n'
                          << "thread_safe_queue<cv::Mat> m_resized_queue -> Size = "
                          << m_resized_queue.unsafe_size() << '\n'
                          << "thread_safe_queue<internal_t> m_after_inference_queue "
                             "-> Size = "
                          << m_after_inference_queue.unsafe_size() << '\n'
                          << "thread_safe_queue<pose_set> m_pose_sets_queue -> Size "
                             "= "
                          << m_pose_sets_queue.unsafe_size() << std::endl;
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(milli * 1ms);
            }
        }));
    }

    ~basic_stream_manager()
    {
        std::unique_lock lk{ m_global_mutex };
        m_shutdown_notifier.wait(lk, [this] { return m_remaining_num == 0; });
        m_shutdown = true;
        m_shutdown_notifier.notify_all();
        for (auto&& x : m_thread_tracer)
            x.get();
    }

    std::atomic<size_t> m_remaining_num{ 0 };

    bool m_shutdown = false;
    std::mutex m_global_mutex;
    std::condition_variable m_shutdown_notifier;
    std::vector<std::future<void>> m_thread_tracer;

    using pose_set = std::vector<human_t>;

    /*
   * Connections:
   * input -> resize.
   * input -> replica.
   * resize -> dnn inference.
   * dnn inference -> post processing.
   * post processing -> visualization + output.
   */

    std::condition_variable m_cv_data_i;
    std::condition_variable m_cv_resize;
    std::condition_variable m_cv_dnn_inf;
    std::condition_variable m_cv_post_processing;

    thread_safe_queue<cv::Mat> m_input_queue;
    thread_safe_queue<cv::Mat> m_input_queue_replica;
    thread_safe_queue<cv::Mat> m_resized_queue;
    thread_safe_queue<internal_t> m_after_inference_queue;
    thread_safe_queue<pose_set> m_pose_sets_queue;

    thread_pool m_thread_pool;
};

template <typename DNNEngine, typename Parser>
class stream {
public:
    stream(DNNEngine& engine, Parser& parser, size_t parser_replica = 4,
        size_t queue_max_size = 64)
        : m_stream_manager(queue_max_size)
        , m_engine_ref(engine)
        , m_main_parser_ref(parser)
        , m_parser_replicas(parser_replica, parser)
    {
        m_parser_refs.push_back(std::ref(parser));
        for (auto&& x : m_parser_replicas)
            m_parser_refs.push_back(std::ref(x));
    }

private:
    basic_stream_manager m_stream_manager;

    DNNEngine& m_engine_ref;
    Parser& m_main_parser_ref;

    std::vector<Parser> m_parser_replicas;
    std::vector<std::reference_wrapper<Parser>> m_parser_refs;
};

template <typename DNNEngine, typename Parser, typename... Others>
stream<DNNEngine, Parser> make_stream(DNNEngine&& engine, Parser&& parser,
    Others&&... others)
{
    return stream<DNNEngine, Parser>(std::forward<DNNEngine>(engine),
        std::forward<Parser>(parser),
        std::forward<Others>(others)...);
}

} // namespace swiftpose