#include <gflags/gflags.h>
#include <stdtracer>

#include "paf.h"
#include "post-process.h"

DEFINE_int32(input_height, 256, "Height of input image.");
DEFINE_int32(input_width, 384, "Width of input image.");
DEFINE_int32(gauss_kernel_size, 13, "Gauss kernel size for smooth operation.");
DEFINE_int32(repeat, 20, "Number of repeats.");

void test_1()
{
    const int h = FLAGS_input_height;
    const int w = FLAGS_input_width;

    peak_finder_t<float> peak_finder(n_joins, h, w, FLAGS_gauss_kernel_size);

    tensor_t<float, 3> heatmap(nullptr, n_joins, h, w);
    const int n = FLAGS_repeat;
    for (int i = 0; i < n; ++i) {
        peak_finder.find_peak_coords(heatmap, 0.05, false);
    }
}

int main(int argc, char *argv[])
{
    TRACE(__func__);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    test_1();
    return 0;
}
