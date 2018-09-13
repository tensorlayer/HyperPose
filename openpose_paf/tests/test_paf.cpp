#include "../paf.h"
#include "../tensor.h"
#include <cstdio>

void test_1()
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
        process_conf_paf(height, width, j + 1, c, conf.data(), paf.data());
    }
}

int main()
{
    TRACE(__func__);
    test_1();
    return 0;
}
