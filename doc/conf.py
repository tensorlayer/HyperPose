import subprocess
subprocess.call('doxygen', shell=True)

project = 'openpose-plus'
copyright = '2018, tensorlayer'
author = 'tensorlayer'

version = ''
release = ''

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    # 'sphinx.ext.pngmath',
    'breathe',
]

templates_path = ['_templates']
source_suffix = '.rst'

master_doc = 'index'

language = None

exclude_patterns = []

pygments_style = 'sphinx'

html_theme = 'alabaster'

html_static_path = [
    # '_static',
]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# Output file base name for HTML help builder.
htmlhelp_basename = 'openpose-plusdoc'

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        'openpose-plus',
        'openpose-plus Documentation',
        [author],
        1,
    ),
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        'openpose-plus',
        'openpose-plus Documentation',
        author,
        'openpose-plus',
        'One line description of project.',
        'Miscellaneous',
    ),
]

todo_include_todos = True

breathe_projects = {
    'openpose-plus': './xml',
}

breathe_default_project = 'openpose-plus'
