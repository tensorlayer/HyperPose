#include "../paf.h"
#include "../tensor.h"
#include "../vis.h"
#include <cstdio>

void test_c()
{
    TRACE(__func__);

    const int height = 46;
    const int width = 54;
    const int j = 18;
    const int c = 19;

    tensor_t<float, 3> conf(nullptr, height, width, j + 1);
    tensor_t<float, 3> paf(nullptr, height, width, c * 2);

    load_idx_file(conf, "conf.idx");
    load_idx_file(paf, "paf.idx");

    int n = 10;
    for (int i = 0; i < n; ++i) {
        TRACE("process_conf_paf::c");
        process_conf_paf(height, width, j + 1, c, conf.data(), paf.data());
    }
}

void test_cpp()
{
    TRACE(__func__);

    const int height = 46;
    const int width = 54;
    const int j = 18;
    const int c = 19;

    tensor_t<float, 3> conf(nullptr, height, width, j + 1);
    tensor_t<float, 3> paf(nullptr, height, width, c * 2);

    load_idx_file(conf, "conf.idx");
    load_idx_file(paf, "paf.idx");

    const cv::Size up_size(8 * width, 8 * height);
    auto p = create(height, width, up_size.height, up_size.width, j + 1, c);
    int n = 10;
    for (int i = 0; i < n; ++i) {
        TRACE("process_conf_paf::c++");
        const auto humans = (*p)(conf.data(), paf.data());
        {
            TRACE("process_conf_paf::c++::draw");
            cv::Mat img(up_size, CV_8UC(3));
            for (const auto h : humans) {
                h.print();
                draw_human(img, h);
            }
            cv::imwrite("result.png", img);
        }
    }
}

int main()
{
    TRACE(__func__);
    test_c();
    test_cpp();
    return 0;
}
