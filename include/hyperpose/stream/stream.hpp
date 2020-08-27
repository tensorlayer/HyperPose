#pragma once

/// \file stream.hpp
/// \brief Stream processing for pose estimation.

#include <future>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "../utility/data.hpp"
#include "../utility/human.hpp"
#include "../utility/thread_pool.hpp"
#include "../utility/thread_safe_queue.hpp"

namespace hyperpose {

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

    basic_stream_manager(size_t uniform_max_size, bool use_original_resolution, bool keep_ratio, cv::Size inp_size);

    size_t processed_num() const noexcept;

    void read_from(const std::vector<cv::Mat>&);
    void read_from(cv::VideoCapture&);
    void read_from(cv::Mat);

    void resize_from_inputs(cv::Size size);

    template <typename Engine>
    void dnn_inference_from_resized_images(Engine&& engine);

    template <typename ParserList>
    void parse_from_internals(ParserList&& parser_list);

    template <typename NameGenerator>
    enable_if_name_getter_t<NameGenerator> write_to(NameGenerator&& name_getter);
    void write_to(cv::VideoWriter&);

    void add_queue_monitor(double milli = 1000);

    ~basic_stream_manager();

private:
    std::atomic<size_t> m_remaining_num{ 0 };
    std::atomic<size_t> m_ingest{ 0 };
    const bool m_use_original_resolution;
    const bool m_keep_ratio;
    cv::Size m_input_size;

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

/// \brief The class to do end-to-end stream processing for pose estimation.
/**
 * @code
 * namespace pp = hyperpose;
 *
 * // Create input stream.
 * cv::VideoCapture cap(...);
 *
 * // Create output stream.
 * cv::VideoWriter writer(...);
 *
 * // Create engine.
 * pp::dnn::tensorrt engine(...);
 *
 * // Create PAF processor.
 * hyperpose::parser::paf paf_processor{};
 *
 * // Create a stream.
 * auto stream = hyperpose::make_stream(engine, paf_processor);
 *
 * // Set input stream asynchronously.
 * stream.async() << cap;
 *
 * // Set the output stream synchronously and waiting for the work done.
 * stream.sync() >> writer.
 *
 * @endcode
 */
/// \tparam DNNEngine The DNN engine class. (e.g. hyperpose::dnn::tensorrt)
/// \tparam Parser The post-processing class. (e.g. hyperpose::parser::paf)
template <typename DNNEngine, typename Parser>
class stream {
public:
    /// \brief Constructor of class stream.
    /// \param engine The reference to the DNN engine object.
    /// \param parser The reference to the parser object.
    /// \param use_original_resolution If true, the output image size will be the input image size, otherwise the DNN input size.
    /// \param keep_ratio Whether to keep original aspect ratio. This is good for accuracy, but requires extra steps to refine the `hyperpose::human_t`.
    /// \param parser_cnt The number of parsers to do parallel post processing. (default: the DNN engine's batch size)
    /// \param queue_max_size The maximum value of internal packet queue sizes.
    /// \note Using the DNN input size as the output resolution(`use_original_resolution = false`) is usually faster.
    /// Because it reduces 1x memory copy. However, the DNN input size are usually much smaller than what you expected.
    /// Hence, you can set it `true` for output image quality, or set it `false` for performance.
    /// \note We highly recommend you to initialize the stream using `hyperpose::make_stream`.
    explicit stream(DNNEngine& engine, Parser& parser, bool use_original_resolution = false, bool keep_ratio = false, size_t parser_cnt = 0, size_t queue_max_size = 128)
        : m_stream_manager(queue_max_size, use_original_resolution, keep_ratio, engine.input_size())
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

    /// This nested class is used for convenient input/output asynchronization.
    class async_handler {
        stream& m_stream;

    public:
        ///
        /// \param s The stream class to bind.
        /// \note We highly recommend you to initialize this class by `stream.async()`.
        async_handler(stream& s)
            : m_stream(s)
        {
        }

        /// Set a input stream asynchronously.
        /// \tparam S The input stream. (e.g., cv::vector<cv::Mat>, cv::VideoCapture, etc.)
        /// \return The object itself.
        template <typename S>
        friend async_handler& operator<<(async_handler&&, S&&);

        /// Set a output stream asynchronously.
        /// \tparam S The output stream. (e.g., cv::VideoWriter)
        /// \return Self.
        template <typename S>
        friend async_handler& operator>>(async_handler&&, S&&);
    };

    /// Synchronized version of async_handler.
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

    /// \brief The asynchronized handler to do data IO.
    /// \return The async_handler binding current stream.
    async_handler async() { return *this; }

    /// \brief The synchronized handler to do data IO.
    /// \return The sync_handler binding current stream.

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

    /// Concurrent monitor to report the status of the queues, remaining records and so on.
    /// \param milli_count The interval to report current stream status.
    void add_monitor(size_t milli_count)
    {
        m_stream_manager.add_queue_monitor(milli_count);
    }

    /// Counting ingested frames.
    /// \return The number of ingested frames.
    size_t processed_num() const noexcept { return m_stream_manager.processed_num(); }

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

/// Yet an easier way to build the stream class.
/// \tparam DNNEngine DNN engine type.
/// \tparam Parser Post-processing parser type.
/// \tparam Others Other parameters.
/// \param engine The reference to the DNN engine object.
/// \param parser The reference to the parser object.
/// \param others Other parameters in `hyperpose::stream`'s constructor.
/// \return The stream object.
/**
 * @code
 * namespace pp = hyperpose;
 *
 * // Create engine.
 * pp::dnn::tensorrt engine(...);
 *
 * // Create PAF processor.
 * hyperpose::parser::paf paf_processor{};
 *
 * // Create a stream.
 * auto stream = hyperpose::make_stream(engine, paf_processor);
 *
 * @endcode
 */
template <typename DNNEngine, typename Parser, typename... Others>
auto make_stream(DNNEngine&& engine, Parser&& parser, Others&&... others)
{
    return stream<
        std::remove_reference_t<DNNEngine>,
        std::remove_reference_t<Parser>>(std::forward<DNNEngine>(engine),
        std::forward<Parser>(parser),
        std::forward<Others>(others)...);
}

} // namespace hyperpose

// Implementation.
namespace hyperpose {

template <typename Engine>
void basic_stream_manager::dnn_inference_from_resized_images(Engine&& engine)
{
    while (true) {
        {
            std::unique_lock lk{ m_resized_queue.m_mu };
            m_cv_resize.wait(lk, [this] { return m_resized_queue.m_size > 0 || m_shutdown; });
        }

        if (m_pose_sets_queue.m_size == 0 && m_shutdown)
            break;

        auto resized_inputs = m_resized_queue.dump(engine.max_batch_size());
        auto internals = engine.inference(std::move(resized_inputs));

        m_after_inference_queue.wait_until_pushed(std::move(internals));

        m_cv_dnn_inf.notify_one();
    }
}

template <typename ParserList>
void basic_stream_manager::parse_from_internals(ParserList&& parser_list)
{
    while (true) {
        {
            std::unique_lock lk{ m_after_inference_queue.m_mu };
            m_cv_dnn_inf.wait(lk,
                [this] { return m_after_inference_queue.m_size > 0 || m_shutdown; });
        }

        if (m_pose_sets_queue.m_size == 0 && m_shutdown)
            break;

        auto internals = m_after_inference_queue.dump_all();

        std::vector<std::future<pose_set>> futures;
        futures.reserve(internals.size());

        for (size_t round_robin = 0; round_robin < internals.size(); ++round_robin) {
            if (round_robin >= parser_list.size())
                futures[round_robin - parser_list.size()].wait();

            futures.push_back(m_thread_pool.enqueue(
                [&parser_list, &internals, round_robin]() -> pose_set {
                    auto poses = parser_list.at(round_robin % parser_list.size()).get().process(std::move(internals[round_robin]));
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
}

template <typename NameGetter>
basic_stream_manager::enable_if_name_getter_t<NameGetter>
basic_stream_manager::write_to(NameGetter&& name_getter)
{
    while (true) {
        {
            std::unique_lock lk{ m_pose_sets_queue.m_mu };
            m_cv_post_processing.wait(lk, [this] { return m_pose_sets_queue.m_size > 0 || m_shutdown; });
        }

        if (m_pose_sets_queue.m_size == 0 && m_shutdown)
            break;

        auto pose_set = m_pose_sets_queue.dump_all();
        for (auto&& poses : pose_set) {
            auto raw_image = m_input_queue_replica.dump().value();
            for (auto&& pose : poses) {
                if (m_keep_ratio)
                    resume_ratio(pose, raw_image.size(), m_input_size);
                draw_human(raw_image, pose);
            }
            cv::imwrite(name_getter(), raw_image);
            --m_remaining_num;
        }

        if (m_remaining_num == 0 && m_pose_sets_queue.m_size == 0)
            break;
    }
    m_shutdown_notifier.notify_one();
}
}