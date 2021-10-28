# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = .build
SPHINXAPIDOC  ?= sphinx-apidoc
SPHINX_APIDOC_OPTIONS = members
SPHINX_APIDOC_TEMPLATEDIR = _templates/apidoc

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile apidoc

apidoc:
	@SPHINX_APIDOC_OPTIONS=$(SPHINX_APIDOC_OPTIONS) $(SPHINXAPIDOC) -f -T -e -M -d 2 -t $(SPHINX_APIDOC_TEMPLATEDIR) -o ./colossalai ../colossalai
# @$(SPHINXAPIDOC) -f -o ./model_zoo ../model_zoo
# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
