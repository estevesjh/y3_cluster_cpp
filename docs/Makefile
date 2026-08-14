# Sphinx makefile. `make html` == the documented
# `sphinx-build -W -b html source build/html` check.
#
# Uses the dedicated docs venv (created automatically on first run) so it
# never depends on whatever sphinx-build happens to be on PATH — the
# system /usr/bin/sphinx-build is too old and lacks the theme.
VENV          ?= $(HOME)/venvs/y3docs
SPHINXBUILD   := $(VENV)/bin/sphinx-build
SPHINXOPTS    ?= -W
SOURCEDIR     = source
BUILDDIR      = build

$(SPHINXBUILD): requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --quiet --upgrade -r requirements.txt
	@touch $(SPHINXBUILD)

help: $(SPHINXBUILD)
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help Makefile

# Keep the catch-all pattern rule below from swallowing these as targets.
requirements.txt: ;

%: Makefile $(SPHINXBUILD)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
