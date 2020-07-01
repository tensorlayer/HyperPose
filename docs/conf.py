# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../Hyperpose'))

# -- Project information -----------------------------------------------------

project = 'HyperPose'
copyright = '2020, Jiawei Liu, Yixiao Guo, Luo Mai, Guo Li, Hao Dong'
author = 'Jiawei Liu, Yixiao Guo, Luo Mai, Guo Li, Hao Dong'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    'sphinx_markdown_tables',
    'recommonmark',
    'numpydoc'
]

autodoc_mock_imports = [
    'gridfs',
    'horovod',
    'hyperdash',
    'imageio',
    'lxml',
    'matplotlib',
    'PIL',
    'progressbar',
    'pymongo',
    'scipy',
    'skimage',
    'sklearn',
    'tensorflow',
    'tqdm',
    'h5py',

    'tensorlayer.third_party.roi_pooling.roi_pooling.roi_pooling_ops',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

master_doc = 'index'

# Do doxygen
import subprocess

subprocess.call('cd doxygen; doxygen Doxyfile', shell=True)
subprocess.call('mkdir -p _build/html', shell=True)
subprocess.call('cp -r doxygen/build/html _build//html/cpp', shell=True)
