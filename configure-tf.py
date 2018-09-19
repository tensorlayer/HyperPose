#!/usr/bin/env python3

import tensorflow as tf

print(tf.sysconfig.get_include())
print(tf.sysconfig.get_lib())


def gen_cmake(filename):
    with open(filename, 'w') as f:
        f.write('# tensorflow=%s\n' % tf.__version__)
        f.write('INCLUDE_DIRECTORIES(%s)\n' % tf.sysconfig.get_include())
        f.write('LINK_DIRECTORIES(%s)\n' % tf.sysconfig.get_lib())


def gen_make(filename):
    with open(filename, 'w') as f:
        f.write('# tensorflow=%s\n' % tf.__version__)
        f.write('TF_INCLUDE_PATH = %s\n' % tf.sysconfig.get_include())
        f.write('TF_LIBRARY_PATH = %s\n' % tf.sysconfig.get_lib())


# gen_make('Makefile.auto')
gen_cmake('src/tensorflow.cmake')
