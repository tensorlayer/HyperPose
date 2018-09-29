#pragma once
#include <npp.h>

#include <cstdio>
#include <cstdlib>

struct npp_status_checker {
    void operator<<(NppStatus status) const
    {
        if (status != NPP_SUCCESS) {
            fprintf(stderr, "NPP Error: %d\n", status);
            exit(1);
        }
    }
};

extern npp_status_checker check_npp_status;
