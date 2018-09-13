#!/usr/bin/env python3
import numpy as np

from openpose_paf import openpose_paf as libpaf


def test_1():
    d = np.load('network-output.npz')
    humans = libpaf.process(d['conf'], d['paf'])

    for h in humans:
        print(h)


test_1()
