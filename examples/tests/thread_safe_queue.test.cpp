#include <hyperpose/utility/thread_safe_queue.hpp>

// Test codes.
int main()
{
    using namespace hyperpose;
#ifdef NDEBUG
    std::cerr << "Debug Flags not set!\n";
#endif
    { // Test 1
        constexpr size_t iters = 10;
        thread_safe_queue<std::vector<int>> queue(iters * 3);
        std::atomic<size_t> answer = 0;

        {
            std::vector<std::future<void>> vec;

            std::vector template1 = { 1, 3, 4 };
            std::vector template2 = { 11, 2, 62, 14, 231 };
            std::vector template3 = { 13, 534, 12, 5, 31, 2 };

            answer += std::accumulate(template1.begin(), template1.end(), 0) * iters;
            answer += std::accumulate(template2.begin(), template2.end(), 0) * iters;
            answer += std::accumulate(template3.begin(), template3.end(), 0) * iters;

            vec.push_back(std::async([&] { queue.push(template1); }));
            vec.push_back(std::async([&] { queue.push(template2); }));
            vec.push_back(std::async([&] { queue.push(template3); }));

            auto f = std::async([&answer, &queue] {
                while (answer != 0) {
                    auto v = queue.dump();
                    if (v.has_value())
                        for (auto&& x : v.value())
                            answer -= x;
                }
            });

            for (size_t i = 0; i < iters - 1; ++i) {
                vec.push_back(std::async([&] { queue.push(template1); }));
                vec.push_back(std::async([&] { queue.push(template2); }));
                vec.push_back(std::async([&] { queue.push(template3); }));
            }

            for (auto&& x : vec)
                x.get();
            f.get();
        }

        assert(answer == 0);
    }

    { // Test 2
        constexpr size_t iters = 10;
        thread_safe_queue<std::vector<int>> queue(iters * 3);
        std::atomic<size_t> answer = 0;

        {
            std::vector<std::future<void>> vec;

            std::vector template1 = { 1, 3, 4 };
            std::vector template2 = { 11, 2, 62, 14, 231 };
            std::vector template3 = { 13, 534, 12, 5, 31, 2 };

            answer += std::accumulate(template1.begin(), template1.end(), 0) * iters;
            answer += std::accumulate(template2.begin(), template2.end(), 0) * iters;
            answer += std::accumulate(template3.begin(), template3.end(), 0) * iters;

            vec.push_back(std::async([&] { queue.push(template1); }));
            vec.push_back(std::async([&] { queue.push(template2); }));
            vec.push_back(std::async([&] { queue.push(template3); }));

            auto f = std::async([&answer, &queue] {
                while (answer != 0) {
                    auto v = queue.dump(queue.capacity());
                    for (auto&& x : v) {
                        for (auto&& y : x)
                            answer -= y;
                    }
                }
            });

            for (size_t i = 0; i < iters - 1; ++i) {
                vec.push_back(std::async([&] { queue.push(template1); }));
                vec.push_back(std::async([&] { queue.push(template2); }));
                vec.push_back(std::async([&] { queue.push(template3); }));
            }

            for (auto&& x : vec)
                x.get();
            f.get();
        }

        assert(answer == 0);
    }

    {
        // Test 3
        thread_safe_queue<int> queue(4);

        queue.push(std::vector{ 1, 2, 3 });
        auto f = std::async([&] {
            queue.wait_until_pushed(std::vector{ 1, 2, 3, 4 });
        });

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(1s);
        auto&& v = queue.dump_all();

        f.get();

        for (int i = 0; i < queue.capacity(); ++i)
            assert(i + 1 == queue.dump().value());
    }
}