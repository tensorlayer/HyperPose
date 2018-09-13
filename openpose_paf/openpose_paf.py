import os
import platform
from ctypes import c_char_p, c_void_p, cdll


def _load_so(name):
    prefix = os.path.dirname(os.path.realpath(__file__))
    suffix = 'so' if platform.uname()[0] != 'Darwin' else 'dylib'
    libpath = '%s/%s.%s' % (prefix, name, suffix)
    lib = cdll.LoadLibrary(libpath)
    return lib


lib = _load_so('libpaf')


def process(conf, paf):
    """Get human from openpose network inference result.

    Parameters:
        conf: the confidence map tensor, must a np.array of shape (H, W, J)
        paf: the part affinity field tensor, must a np.array of shape (H, W, 2C)
    """
    h, w, j = conf.shape
    _, _, c = paf.shape
    if (paf.shape[:2] != (h, w)):
        raise ValueError('invalid input')
    c /= 2

    assert (j == 19)
    assert (c == 19)

    lib.process_conf_peak_paf(
        h,
        w,
        j,
        j,
        c,
    )
    # print(conf)
    # print(paf)
    humans = []
    return humans
