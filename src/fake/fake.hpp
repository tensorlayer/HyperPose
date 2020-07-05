#pragma once

inline void error_exit_fake()
{
    std::cerr << "Using a fake library!" << std::endl;
    std::exit(-1);
}