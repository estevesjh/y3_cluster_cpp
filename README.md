# y3_cluster_cpp

High-performance implementation of the cluster-population and
cluster-lensing calculations used in the **DES cluster-cosmology
analysis**: C++ (and CUDA) CosmoSIS modules that predict cluster number
counts and stacked lensing profiles in bins of observed richness and
redshift, together with the optical selection-bias and projection-lensing
model framework of [Costanzi et al. 2026](https://arxiv.org/abs/2604.05833).

This repository is the **cluster-observable prediction engine**. The
complete analysis also uses CosmoSIS Standard Library modules, local
Python modules (`y3_buzzard/`), calibration data (`data/`), and
run-management configurations in sibling repositories (`add-names-here`).

## Documentation

Full technical documentation (scientific model, numerical recipes,
module reference, validation) lives under `docs/source/` and is hosted as
a private Read the Docs project. To build it locally:

```bash
python3 -m venv ~/venvs/y3docs && source ~/venvs/y3docs/bin/activate
pip install -r docs/requirements.txt
sphinx-build -W -b html docs/source docs/build/html
```

## Installation

The pipeline builds only on NERSC Perlmutter with a specific non-default
toolchain — see [BUILDING.md](BUILDING.md) for the full recipe. In short:

```bash
# on a Perlmutter GPU node, after the environment setup in BUILDING.md
cd $Y3_CLUSTER_CPP_DIR && mkdir -p release-build && cd release-build
cmake -DUSE_CUDA=On -DY3GCC_TARGET_ARCH=80-real ... -G Ninja $Y3_CLUSTER_CPP_DIR
ninja
ctest -j 10
```
<div class="alert alert-block alert-info">
<b>Question:</b> It also runs on Artemis. Don’t you want add this info here?
</div>

## Minimal example

With the modules built and the CosmoSIS environment active, run the
single-sample smoke pipeline (predicts number counts, one-halo shear, and
projection shear at the fiducial point using the `src/pipelines/des_y3`
`fast_mass` C++ backends):

```bash
cd $Y3_CLUSTER_CPP_DIR
cosmosis cosmosis-models/des_y3.ini
```

This run will take ~X s/min/h and the outputs are saved in `des_y3_test_output/`. See the docs'
[Running the reference pipeline](docs/source/running.md) page for the
full module trace, environment setup, and a caveat about wiring this
into an end-to-end likelihood evaluation. The production MCMC pipeline
(DES Y1-generation modules) lives in the `des-cluster-nersc` repository.
