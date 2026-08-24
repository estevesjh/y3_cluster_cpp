# Reproducible build: every figure, table and value is script-generated.
# values.tex / figs / tables come from ../07_compare.py (which consumes
# ../outputs/*.npz produced by ../run_all.sh); downstream.tex is collected
# by scripts/collect_downstream.py (runs the shear1h2h validator and the
# halo_model test suite). Nothing under figs/, tables/, values.tex or
# build/ is ever hand-edited.

CONDA    := /opt/homebrew/Caskroom/miniforge/base/envs/y3cl_je_macos/bin/python
SECTIONS := $(wildcard sections/*.tex)
NPZ      := $(wildcard ../outputs/*.npz)

all: build/report.pdf

values.tex: ../07_compare.py $(NPZ)
	cd .. && $(CONDA) 07_compare.py

build/downstream.tex: scripts/collect_downstream.py
	mkdir -p build
	$(CONDA) scripts/collect_downstream.py

build/report.pdf: report.tex $(SECTIONS) values.tex build/downstream.tex
	mkdir -p build
	pdflatex -interaction=nonstopmode -output-directory=build report.tex
	pdflatex -interaction=nonstopmode -output-directory=build report.tex

clean:
	rm -rf build/*

.PHONY: all clean
