#!/usr/bin/env python3

import numpy as np
import scipy.stats as st

def _normalize(t):
    return t / t.sum()


def _gauss_kernel_1d(ksize, nsig):
    interval = (2 * nsig + 1.) / ksize
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., ksize + 1)
    y = np.diff(st.norm.cdf(x))
    return y

# return _normalize(np.sqrt(np.outer(y, y)))


gk =_gauss_kernel_1d(17, 3.0)

print('%s' %(', '.join('%f'%x for x in gk)))