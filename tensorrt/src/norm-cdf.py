#!/usr/bin/env python3

import numpy as np
import scipy.stats as st


def _gauss_kernel_1d(ksize, nsig):
    interval = (2 * nsig + 1.) / ksize
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., ksize + 1)
    y = np.diff(st.norm.cdf(x))
    return y


ksize = 9
gk = _gauss_kernel_1d(ksize, 3.0)

print('{%s};' % ','.join(str(x) for x in gk))
