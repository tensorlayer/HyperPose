default: sphinx-doc

doxygen-doc:
	doxygen

breathe-doc: doxygen-doc
	breathe-apidoc -o output xml


SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = output
BUILDDIR      = build

# TODO: make it work
sphinx-doc: breathe-doc
	$(SPHINXBUILD) -c . "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
