#pragma once

/// \file stream.hpp
/// \brief Stream processing for pose estimation.
/// \author Jiawei Liu(github.com/ganler)

#include <future>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../utility/data.hpp"
#include "../utility/thread_pool.hpp"
#include "../utility/thread_safe_queue.hpp"
#include "../utility/viz.hpp"

namespace poseplus {

// FIXME: DEADLOCK!

class basic_stream_manager {
private:
    template <typename NameGetter>
    using enable_if_name_getter_t = std::enable_if_t<
        std::is_convertible_v<
            std::result_of_t<decltype (&NameGetter::operator())()>,
            std::string>>;

public:
    template <typename, typename>
    friend class stream;

    basic_stream_manager(size_t uniform_max_size);

    void read_from(const std::vector<cv::Mat>&);
    void read_from(cv::VideoCapture&);
    void read_from(cv::Mat);

    void resize_from_inputs(cv::Size size);

    template <typename Engine>
    void dnn_inference_from_resized_images(Engine&& engine);

    template <typename ParserList>
    void parse_from_internals(ParserList&& parser_list);

    template <typename NameGetter>
    enable_if_name_getter_t<NameGetter> write_to(NameGetter&& name_getter);
    void write_to(cv::VideoWriter&);

    void add_queue_monitor(double milli = 1000);

    ~basic_stream_manager();

private:
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
    stream(DNNEngine& engine, Parser& parser, size_t parser_cnt = 0, size_t queue_max_size = 128)
        : m_stream_manager(queue_max_size)
        , m_engine_ref(engine)
        , m_main_parser_ref(parser)
        , m_parser_replicas(parser_cnt == 0 ? engine.max_batch_size() : parser_cnt, parser)
    {
        m_parser_refs.reserve(m_parser_replicas.size() + 1);
        m_parser_refs.push_back(std::ref(parser));
        for (auto&& x : m_parser_replicas)
            m_parser_refs.push_back(std::ref(x));
        build_internal_running_graph();
    }

    class async_handler {
        stream& m_stream;

    public:
        async_handler(stream& s)
            : m_stream(s)
        {
        }
        template <typename S>
        friend async_handler& operator<<(async_handler&&, S&&);
        template <typename S>
        friend async_handler& operator>>(async_handler&&, S&&);
    };

    class sync_handler {
        stream& m_stream;

    public:
        sync_handler(stream& s)
            : m_stream(s)
        {
        }
        template <typename S>
        friend sync_handler& operator<<(sync_handler&&, S&&);
        template <typename S>
        friend sync_handler& operator>>(sync_handler&&, S&&);
    };

    async_handler async() { return *this; }
    sync_handler sync() { return *this; }

    template <typename S>
    friend async_handler& operator<<(async_handler&& handler, S&& source)
    {
        handler.m_stream.get_tracer().push_back(handler.m_stream.add_input_stream(source));
        return handler;
    }

    template <typename S>
    friend async_handler& operator>>(async_handler&& handler, S&& source)
    {
        handler.m_stream.get_tracer().push_back(handler.m_stream.add_output_stream(source));
        return handler;
    }

    template <typename S>
    friend sync_handler& operator<<(sync_handler&& handler, S&& source)
    {
        handler.m_stream.add_input_stream(source);
        return handler;
    }

    template <typename S>
    friend sync_handler& operator>>(sync_handler&& handler, S&& source)
    {
        handler.m_stream.add_output_stream(source);
        return handler;
    }

    void add_monitor(size_t milli_count)
    {
        m_stream_manager.add_queue_monitor(milli_count);
    }

private:
    auto& get_tracer()
    {
        return m_stream_manager.m_thread_tracer;
    }

    template <typename S>
    auto add_input_stream(S&& s)
    {
        // Only 1 input stream at the same time.
        return m_mpsc_worker.enqueue([this, &s] {
            m_stream_manager.read_from(std::forward<S>(s));
        });
    }

    template <typename S>
    auto add_output_stream(S&& s)
    {
        return std::async([this, &s] {
            m_stream_manager.write_to(std::forward<S>(s));
        });
    }

    void build_internal_running_graph()
    {
        auto& tracer = m_stream_manager.m_thread_tracer;

        tracer.push_back(std::async([this] {
            m_stream_manager.resize_from_inputs(m_engine_ref.input_size());
        }));

        tracer.push_back(std::async([this] {
            m_stream_manager.dnn_inference_from_resized_images(m_engine_ref);
        }));

        tracer.push_back(std::async([this] {
            m_stream_manager.parse_from_internals(m_parser_refs);
        }));
    }

private:
    basic_stream_manager m_stream_manager;

    DNNEngine& m_engine_ref;
    Parser& m_main_parser_ref;
    thread_pool m_mpsc_worker = thread_pool(1);

    std::vector<Parser> m_parser_replicas;
    std::vector<std::reference_wrapper<Parser>> m_parser_refs;
};

template <typename DNNEngine, typename Parser, typename... Others>
auto make_stream(DNNEngine&& engine, Parser&& parser, Others&&... others)
{
    return stream<
        std::remove_reference_t<DNNEngine>,
        std::remove_reference_t<Parser>>(std::forward<DNNEngine>(engine),
        std::forward<Parser>(parser),
        std::forward<Others>(others)...);
}

} // namespace poseplus

// Implementation.
namespace poseplus {

template <typename Engine>
void basic_stream_manager::dnn_inference_from_resized_images(Engine&& engine)
{
    while (true) {
        //        std::cout << __PRETTY_FUNCTION__ << " LOCK\n";
        {
            std::unique_lock lk{ m_resized_queue.m_mu };
            m_cv_resize.wait(lk, [this] { return m_resized_queue.m_size > 0 || m_shutdown; });
        }
        //        std::cout << __PRETTY_FUNCTION__ << " UNLOCK\n";

        if (m_pose_sets_queue.m_size == 0 && m_shutdown)
            break;

        auto resized_inputs = m_resized_queue.dump(engine.max_batch_size());
        auto internals = engine.inference(std::move(resized_inputs));

        m_after_inference_queue.wait_until_pushed(std::move(internals));

        m_cv_dnn_inf.notify_one();
    }
    //    std::cout << "Exit: " << __PRETTY_FUNCTION__ << std::endl;
}

template <typename ParserList>
void basic_stream_manager::parse_from_internals(ParserList&& parser_list)
{
    while (true) {
        //        std::cout << __PRETTY_FUNCTION__ << " LOCK\n";
        {
            std::unique_lock lk{ m_after_inference_queue.m_mu };
            m_cv_dnn_inf.wait(lk,
                [this] { return m_after_inference_queue.m_size > 0 || m_shutdown; });
        }
        //        std::cout << __PRETTY_FUNCTION__ << " UNLOCK\n";

        if (m_pose_sets_queue.m_size == 0 && m_shutdown)
            break;

        auto internals = m_after_inference_queue.dump_all();

        std::vector<std::future<pose_set>> futures;
        futures.reserve(internals.size());

        for (size_t round_robin = 0; round_robin < internals.size(); ++round_robin) {
            if (round_robin >= parser_list.size())
                futures[round_robin - parser_list.size()].wait();

            futures.push_back(m_thread_pool.enqueue(
                [&parser_list, &internals, rb = round_robin]() -> pose_set {
                    auto poses = parser_list.at(rb % parser_list.size()).get().process(std::move(internals[rb]));
                    return poses;
                }));
        }

        std::vector<pose_set> pose_sets{};
        pose_sets.reserve(futures.size());

        for (auto&& f : futures)
            pose_sets.push_back(f.get());

        m_pose_sets_queue.wait_until_pushed(std::move(pose_sets));
        m_cv_post_processing.notify_one();
    }
    //    std::cout << "Exit: " << __PRETTY_FUNCTION__ << std::endl;
}

template <typename NameGetter>
basic_stream_manager::enable_if_name_getter_t<NameGetter>
basic_stream_manager::write_to(NameGetter&& name_getter)
{
    while (true) {
        //        std::cout << __PRETTY_FUNCTION__ << " LOCK\n";
        {
            std::unique_lock lk{ m_pose_sets_queue.m_mu };
            m_cv_post_processing.wait(lk, [this] { return m_pose_sets_queue.m_size > 0 || m_shutdown; });
        }
        //        std::cout << __PRETTY_FUNCTION__ << " UNLOCK\n";

        if (m_pose_sets_queue.m_size == 0 && m_shutdown)
            break;

        auto pose_set = m_pose_sets_queue.dump_all();
        for (auto&& poses : pose_set) {
            auto&& raw_image = m_input_queue_replica.dump().value();
            std::cout << poses.size() << std::endl;
            for (auto&& pose : poses)
                draw_human(raw_image, pose);
            cv::imwrite(name_getter(), raw_image);
            --m_remaining_num;
        }

        if (m_remaining_num == 0 && m_pose_sets_queue.m_size == 0)
            break;
    }
    m_shutdown_notifier.notify_one();
    //    std::cout << "Exit: " << __PRETTY_FUNCTION__ << std::endl;
}
}