#pragma once

#include <future>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "../utility/thread_safe_queue.hpp"

namespace swiftpose
{

class basic_stream_manager {
public:
    ~basic_stream_manager() = default;

    /*
     * reader -> [cv::Mat] ->
     * resizer -> [cv::Mat] ->
     * batching_solver -> [buffer] ->
     * Engine -> [internal_t] ->
     * post_processing -> [output_t] -> exporter
     */

    template <typename T>
    std::future<void> setup_reader(T&);

    template <typename T>
    std::future<void> setup_exporter(T&);
protected:
    void setup_resizer(cv::Size);
    void setup_batching_solver(size_t batch_size);

    std::vector<std::future<void>> m_unresolved_futures;
};

template <typename DNN, typename Parser>
class stream
{  // TODO: IMPL.
  public:
  private:

};

}  // namespace swiftpose