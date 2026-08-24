# Building `y3_cluster_cpp` on macOS (CPU-only)

## Scope

This is a from-scratch, **CPU-only** install path for Apple Silicon (arm64)
macOS, verified end-to-end on macOS 26 / M-series. It mirrors
{doc}`installation` (the Perlmutter recipe in `BUILDING.md`) but swaps the
Cray/NERSC toolchain for Homebrew + a pinned `mamba` environment, and skips
CUDA entirely.

```{admonition} Why this exists
:class: note
The pipeline's C++/CUBA integrals and the Python selection/HOD layer are
fast enough now that a laptop is a legitimate place to iterate — no GPU
node, no queue. `USE_CUDA=OFF` was never optional here: no Mac has a
CUDA-capable GPU (Apple Silicon has none at all; Intel Macs dropped NVIDIA
driver support years ago). PAGANI/`gpuintegration` is reference-only
anyway (see {doc}`pipeline_organization`), so this loses nothing that
matters for day-to-day fixed-GL (`0d`, formerly `fast_mass`) work.
```

Version pins below match the NERSC `y3cl_je` conda env as of 2026-08-13
(Python 3.9, GSL 2.7, cfitsio 4.2.0, fftw 3.3.10, cmake 3.26.3, ninja
1.12.1, mpich 4.2.3). Repo commits: this tree's own `master`/working
branch (not a separate pin — build in place), `cosmosis` @
`bba5bed52ea8830f82cdcca0108b584bb368d8c5`, `cosmosis-standard-library` @
`29573689d072ca9e0940f03e597deaf0f4db2d99`, `cuba` @
`57597e36b107c5a45fc2af06bf8872e474de1f9f`, `cubacpp` @
`91b1ec5d4235b83149aed16103c52431f3079cc7`.

## 1. Homebrew tools + Miniforge

```bash
xcode-select --install        # Xcode Command Line Tools, if not already present
# install Homebrew from https://brew.sh if needed

brew install ninja pkg-config
brew install --cask miniforge  # gives you `conda`/`mamba` at
                                # /opt/homebrew/Caskroom/miniforge/base
```

## 2. Pinned mamba environment

```bash
eval "$(/opt/homebrew/Caskroom/miniforge/base/bin/conda shell.bash hook)"

mamba create -n y3cl_je_macos -c conda-forge -y \
  python=3.9 gsl=2.7 cfitsio=4.2.0 fftw=3.3.10 cmake=3.26.3 ninja=1.12.1 \
  mpich=4.2.3 mpi4py=4.0.2 numpy=1.26.4 scipy=1.13.1 astropy=6.0.1 \
  matplotlib=3.9.4 pyyaml=6.0.2 scikit-learn=1.6.1 threadpoolctl=3.5.0 \
  urllib3=2.3.0 zeus-mcmc=2.5.4 liblapack blas \
  c-compiler cxx-compiler fortran-compiler

conda activate y3cl_je_macos

pip install --no-input camb==1.4.0 emcee==3.1.4 future==0.18.3 mpmath numba colossus
pip install --no-input pybind11 dulwich dynesty "nautilus-sampler>=0.6" || true  # optional samplers
```

`cluster_toolkit` (Tinker bias, ξ_NL, halo-model lensing tables in
`y3_buzzard/halo_model_cosmosis.py`) isn't on PyPI/conda — build the
[marcpaterno fork](https://github.com/marcpaterno/cluster_toolkit) from
source, same as the Perlmutter recipe:

```bash
git clone https://github.com/marcpaterno/cluster_toolkit.git /tmp/cluster_toolkit
cd /tmp/cluster_toolkit
GSL_DIR="$CONDA_PREFIX" python setup.py install
```

```{admonition} numba / colossus / mpmath aren't in the NERSC pin list
:class: note
The `y3cl_je` env doesn't need them because the modules that import them
(`sel_function.py`'s `numba.njit`, `halo_model.test.py`'s `colossus`
cross-check, `des_y3_pipeline.test.py`'s `mpmath` NFW check) predate or
postdate whatever snapshot a given `y3cl_je_macos` recipe was pinned
from. Install them explicitly — `ctest` will tell you exactly which
one is missing (`ModuleNotFoundError`) if you skip this.
```

## 3. The compiler-override gotcha — read this first

`c-compiler`/`cxx-compiler`/`fortran-compiler` install `clang`/`clang++`/
`gfortran` **into the conda env** (`$CONDA_PREFIX/bin/clang`, matching the
compilers GSL/fftw/mpich were themselves built with), but on this
platform/channel combination **no `activate.d` script exports `CC`/`CXX`/
`FC` to point at them**. If your shell profile already exports
`CC=/opt/homebrew/opt/llvm/bin/clang` (a common Homebrew-llvm habit), that
leaks straight through `conda activate` and you silently build everything
against the wrong compiler — defeating the entire point of matching
conda-forge's own toolchain (risk: ABI/`libgfortran` mismatches between
Homebrew-built and conda-forge-built objects).

**Always set these explicitly, in every shell, before building anything:**

```bash
export CC="${CONDA_PREFIX}/bin/clang"
export CXX="${CONDA_PREFIX}/bin/clang++"
export FC="${CONDA_PREFIX}/bin/gfortran"
```

## 4. Clone cosmosis / CSL / cuba / cubacpp at pinned commits

```bash
export TOP_DIR="$HOME/cosmosis_y3"
export INTEGRATION_TOOLS_DIR="${TOP_DIR}/y3_pipe_under"
mkdir -p "$TOP_DIR" "$INTEGRATION_TOOLS_DIR"

git clone https://github.com/annis/cosmosis.git "${TOP_DIR}/cosmosis"
git -C "${TOP_DIR}/cosmosis" checkout bba5bed52ea8830f82cdcca0108b584bb368d8c5

git clone https://github.com/annis/cosmosis-standard-library "${TOP_DIR}/cosmosis-standard-library"
git -C "${TOP_DIR}/cosmosis-standard-library" checkout 29573689d072ca9e0940f03e597deaf0f4db2d99

git clone https://github.com/marcpaterno/cuba.git "${INTEGRATION_TOOLS_DIR}/cuba"
git -C "${INTEGRATION_TOOLS_DIR}/cuba" checkout 57597e36b107c5a45fc2af06bf8872e474de1f9f

git clone https://bitbucket.org/mpaterno/cubacpp.git "${INTEGRATION_TOOLS_DIR}/cubacpp"
git -C "${INTEGRATION_TOOLS_DIR}/cubacpp" checkout 91b1ec5d4235b83149aed16103c52431f3079cc7
```

```{admonition} Do not clone a fresh y3_cluster_cpp
:class: warning
Build **this working tree** (the one you already have checked out), not a
separate pinned clone — it carries the current des_y3/Costanzi-2026 work,
which is ahead of any historical pin. `CMakePresets.json`'s `macos-cpu`
preset (added in this repo) targets an in-place build under
`release-build/`, same as the Perlmutter recipe.
```

## 5. Patch CSL's camb interface

A small, vendored patch to `boltzmann/camb/camb_interface.py` (drops a
`sigma_r` computation) lives at `docs/patches/csl_camb_interface.patch`
in this repo:

```bash
CSL_DIR="${TOP_DIR}/cosmosis-standard-library"
git -C "$CSL_DIR" apply "$Y3_CLUSTER_CPP_DIR/docs/patches/csl_camb_interface.patch"
```

Only needed if you're running the CAMB-based `boltzmann/camb` module —
this pipeline's `cp_camb` CosmoPower-emulator path doesn't touch it.

## 6. Build CUBA

```bash
CUBA_DIR="${INTEGRATION_TOOLS_DIR}/cuba"
cd "$CUBA_DIR"
CC="$CC" ./configure
make
```

`FindCUBA.cmake` (from `cubacpp/cmake/modules/`) expects `cuba.h` under
`$CUBA_DIR/include/` and `libcuba.a` under `$CUBA_DIR/lib/` — but the
plain autotools build drops both at `$CUBA_DIR`'s root. Symlink them into
place:

```bash
mkdir -p "$CUBA_DIR/include" "$CUBA_DIR/lib"
ln -sf ../cuba.h "$CUBA_DIR/include/cuba.h"
ln -sf ../config.h "$CUBA_DIR/include/config.h"
ln -sf ../libcuba.a "$CUBA_DIR/lib/libcuba.a"
```

## 7. Build cosmosis core (+ the multinest Fortran fix)

```bash
export GSL_INC="${CONDA_PREFIX}/include"
export GSL_LIB="${CONDA_PREFIX}/lib"
export CFITSIO_DIR="${CONDA_PREFIX}"
export CFITSIO_INC="${CONDA_PREFIX}/include"
export CFITSIO_LIB="${CONDA_PREFIX}/lib"
export FFTW_INCLUDE_DIR="${CONDA_PREFIX}/include"
export FFTW_LIBRARY="${CONDA_PREFIX}/lib"
export LAPACK_LINK="-L${CONDA_PREFIX}/lib -llapack -lblas"
export COSMOSIS_SRC_DIR="${TOP_DIR}/cosmosis/cosmosis"
export COSMOSIS_STANDARD_LIBRARY="${TOP_DIR}/cosmosis-standard-library"
export PYTHONPATH="${TOP_DIR}/cosmosis:${PYTHONPATH:-}"
export PATH="${TOP_DIR}/cosmosis/bin:${PATH}"
export MPIFC=mpif90
export COSMOSIS_ALT_COMPILERS=1
export COSMOSIS_OMP=1
export OMP_NUM_THREADS=4
export USER_FFLAGS="-fallow-argument-mismatch"   # see below
```

`USER_FFLAGS` is the hook `config/compilers.mk` reads for exactly this
kind of situation, but it isn't enough on its own: the bundled multinest
Fortran source has a genuine bug (not a compiler-strictness false
positive) that gfortran ≥ 10 refuses to compile at all. In
`cosmosis/cosmosis/samplers/multinest/multinest_src/nested.f90`, `ic_n`
is declared `integer ic_n(1)` (a rank-1 array), and one `MPI_BCAST` call
passes the raw array expression `ic_n+1` where a scalar `INTEGER` count
is required:

```fortran
! before (rejected by gfortran >= 10's stricter generic-interface check):
call MPI_BCAST(ic_done(0:ic_n(1)),ic_n+1,MPI_LOGICAL,0,MPI_COMM_WORLD,errcode)
! after (the clearly-intended scalar):
call MPI_BCAST(ic_done(0:ic_n(1)),ic_n(1)+1,MPI_LOGICAL,0,MPI_COMM_WORLD,errcode)
```

This slipped through for years because older/looser MPI Fortran
interfaces didn't enforce the mismatch — it's a real bug in the vendored
code, exposed by newer toolchains, not a macOS quirk. Apply that one-line
fix, then build:

```bash
cd "${TOP_DIR}/cosmosis"
python setup.py build
```

The Minuit sampler will fail to compile (`Minuit2/FCNBase.h` not found) —
that's expected and non-fatal (`make` treats it as an ignorable error);
Minuit just won't be available. `libchord.so`/`libchord_mpi.so`
(PolyChord, MPI variant included) and multinest should both build clean.

## 8. Build the CSL modules this pipeline needs

Only build what your `.ini` actually loads — building the entire CSL is
unnecessary and more likely to fail on modules you don't need.

```bash
CSL_DIR="${TOP_DIR}/cosmosis-standard-library"
for mod in utility/consistency mass_function/mf_tinker structure/growth_factor; do
  make -C "${CSL_DIR}/${mod}"
done
```

(`utility/consistency` is pure Python — `make` there is a no-op, that's
fine. Add `boltzmann/camb`, `utility/sample_sigma8`, `boltzmann/sigma_cpp`
too if your pipeline uses CAMB directly instead of `cp_camb`'s CosmoPower
emulator.)

## 9. Configure + build y3_cluster_cpp

```bash
export CUBA_CPP_DIR="${INTEGRATION_TOOLS_DIR}/cubacpp"
export CUBA_DIR="${INTEGRATION_TOOLS_DIR}/cuba"
export Y3_CLUSTER_CPP_DIR="/path/to/this/working/tree"

cd "$Y3_CLUSTER_CPP_DIR"
cmake --preset macos-cpu
cmake --build --preset macos-cpu
```

The `macos-cpu` preset (`CMakePresets.json`) sets `USE_CUDA=OFF` and wires
`CC`/`CXX`/`CONDA_PREFIX`/`CUBA_DIR`/`CUBACPP_DIR`/MPI wrappers from the
environment above — no `Y3GCC_TARGET_ARCH`, no PAGANI.

````{admonition} If ctest picks up the wrong Python
:class: tip
`find_package(Python3)` resolves whatever's first on `PATH` at *configure*
time — if a stray venv shadows the conda env (check with
`cmake -LA | grep Python3_EXECUTABLE`), the Python-based ctest targets
(`des_y3_pipeline_python_test`, `sel_function_test`, `halo_model_test`)
will `ModuleNotFoundError` on packages that are clearly installed. Fix
without a full reconfigure:
```bash
cd release-build
cmake -DPython3_EXECUTABLE="${CONDA_PREFIX}/bin/python3.9" .
```
````

## 10. ctest

```bash
export COSMOSIS_SRC_DIR="${TOP_DIR}/cosmosis/cosmosis"
export PYTHONPATH="${TOP_DIR}/cosmosis:${PYTHONPATH:-}"
cd release-build
ctest --output-on-failure
```

Expect **59/60 passing**. The one failure is `sel_function_test`'s three
deliberately-red tests pinning the known HOD-normalization defect (see
`docs/known_issues/hod_normalization_defect.md` at the repo root — same failure you'd
see on any platform, not macOS-specific).

## 11. Environment script for new shells

```bash
cat > "${TOP_DIR}/cosmosis_init_macos.sh" <<'EOF'
export TOP_DIR=$HOME/cosmosis_y3
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}
export INTEGRATION_TOOLS_DIR=${TOP_DIR}/y3_pipe_under
export CUBA_DIR=${INTEGRATION_TOOLS_DIR}/cuba
export CUBA_CPP_DIR=${INTEGRATION_TOOLS_DIR}/cubacpp

export Y3_CLUSTER_CPP_DIR=/path/to/this/working/tree
export Y3_CLUSTER_WORK_DIR=${Y3_CLUSTER_CPP_DIR}/release-build
export COSMOSIS_SRC_DIR=${COSMOSIS_REPO_DIR}/cosmosis
export PYTHONPATH=${COSMOSIS_REPO_DIR}:${Y3_CLUSTER_CPP_DIR}:${PYTHONPATH:-}
export PATH=${COSMOSIS_REPO_DIR}/bin:${PATH}
export OMP_NUM_THREADS=4
export COSMOSIS_OMP=1

eval "$(/opt/homebrew/Caskroom/miniforge/base/bin/conda shell.bash hook)"
conda activate y3cl_je_macos

# the env's own toolchain -- see "the compiler-override gotcha" above
export CC="${CONDA_PREFIX}/bin/clang"
export CXX="${CONDA_PREFIX}/bin/clang++"
export FC="${CONDA_PREFIX}/bin/gfortran"

cd "${Y3_CLUSTER_CPP_DIR}"
echo "CosmoSIS + y3_cluster_cpp (macOS, CPU-only) initialized."
EOF
```

Source it in every new shell: `source ~/cosmosis_y3/cosmosis_init_macos.sh`.

## Known gaps

- **No CUDA, no PAGANI.** All `*Gpu.so` modules and the `gpuintegration`
  reference backends are unavailable — expected, not a bug. The
  fixed-GL C++ path (this project's actual production speed target)
  never needed them.
- **CosmoSIS `.ini` pipelines live in sibling repos**, not this tree (see
  the top-level `CLAUDE.md`). Running an actual pipeline additionally
  needs `DES_CLUSTER_NERSC_DIR` (or whichever sibling repo's values/data
  files the `.ini` references) and `Y3_CLUSTER_CPP_DIR` on `PYTHONPATH`
  (for `y3_buzzard.*` imports).
- **Legacy Y1-era modules** (`gt_card_cpu`, `mass_y1`, `y1_analysis`) had
  a pre-existing undefined-symbol bug (`INT_LC_LT_DES_t`'s static lookup
  tables were never wired into the `models` library) that Linux's laxer
  `.so` linking silently tolerated. Already fixed upstream in this repo
  (`src/models/CMakeLists.txt`) — nothing to do on a current clone.

## Fetching data files that aren't in git

Some sibling repos deliberately keep large trained artifacts out of git
(e.g. `camb-emulator`'s CosmoPower `.npz` emulator weights — see that
repo's own `.gitignore`/commit messages). If a pipeline module fails to
find one of these, it needs pulling from wherever it's actually stored
(NERSC `$PSCRATCH` in this project's case) via `rsync`/`scp` over your
existing NERSC SSH config — there's nothing macOS-specific here beyond
"this file lives on a remote filesystem, go get it."

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CC`/`CXX` point at `/opt/homebrew/opt/llvm/...` | Shell profile pre-exports them, leaking past `conda activate` | Explicitly re-export to `$CONDA_PREFIX/bin/{clang,clang++,gfortran}` (step 3) |
| `find_library` can't find `libcuba` / `cuba.h` not found during `cmake` | Autotools `cuba` build drops artifacts at `$CUBA_DIR` root, not `lib/`/`include/` | Symlink them (step 6) |
| `no specific subroutine for the generic 'mpi_bcast'` building multinest | Real bug in vendored `nested.f90` (array passed where MPI expects a scalar count), not a flag issue | `ic_n+1` → `ic_n(1)+1` (step 7); `-fallow-argument-mismatch` alone does not fix this one |
| `unrecognized instruction mnemonic` in `externals/catch2/catch.hpp` around `CATCH_TRAP` | Vendored Catch2's `CATCH_PLATFORM_MAC` branch hardcodes x86 `int $3` inline asm | Already fixed in this repo (arch-guarded, falls back to `raise(SIGTRAP)` on non-x86) |
| `ModuleNotFoundError` for `numba`/`colossus`/`mpmath`/`cluster_toolkit` in `ctest` | Not in the base NERSC pin list | Install explicitly (step 2) |
| Python ctest targets `ModuleNotFoundError: cosmosis` despite it being installed | `find_package(Python3)` picked up a stray venv at configure time | `cmake -DPython3_EXECUTABLE=... .` (step 9) |
| `y3_buzzard` / `des_y3` imports fail when running an actual `.ini` | `Y3_CLUSTER_CPP_DIR` not on `PYTHONPATH` | `export PYTHONPATH=$Y3_CLUSTER_CPP_DIR:$PYTHONPATH` |
