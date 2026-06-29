# Building `y3_cluster_cpp` on Perlmutter

## TOC

* [Before you begin](#before-you-begin)
* [Just set up to run an already-built `y3_cluster_cpp`](#just-set-up-to-run-an-already-built-y3_cluster_cpp)
* [But I need to checkout and compile `y3_cluster_cpp`](#but-i-need-to-checkout-and-compile-y3_cluster_cpp)
* [But I have to make CosmoSIS and CosmoSIS Standard Library](#but-i-have-to-make-cosmosis-and-cosmosis-standard-library)
* [I want to make my own Conda environment](#i-want-to-make-my-own-conda-environment)
* [Building the whole stack from scratch](#building-the-whole-stack-from-scratch)

These instructions set you up to use the GPU-enabled version of the
`y3_cluster_cpp` analysis pipeline on NERSC Perlmutter.

> **THIS BUILD IS FRAGILE.** The module versions, PATH layout, and patched
> external headers below are *all* required. Deviate at your own risk to
> your sanity. If you make code changes, follow a clean git workflow — see
> [Marc's notes](https://github.com/marcpaterno/devel-notes/blob/master/git-workflow.md).
> For the CosmoSIS / CSL forks we use the `y3_cluster_cpp_main` branches as
> "main" — **do NOT push to upstream `cosmosis` / `cosmosis-standard-library`
> main.**

## Before you begin

Make sure your shell environment doesn't carry settings that will fight the
install. If any of the following are defined, remove the lines that set
them from `.bashrc` / `.profile` *before* logging back in — un-setting them
mid-session is unreliable, especially under MPI or batch jobs.

1. **`CONDA_PREFIX`** — must be unset (no active conda env).
2. **`VIRTUAL_ENV`** — must be unset (no active Python venv).
3. **`PYTHONPATH`** — must be unset (subverts the venv-based environment
   management we use).
4. **`cosmosis` already on PATH** — `type cosmosis` should print
   `bash: type: cosmosis: not found`.
5. **`LD_LIBRARY_PATH`** — beware. Setting it can cause the runtime
   linker to find the wrong libraries; on Perlmutter it can also cause
   link-time failures.
6. **`~/homebrew/bin` on PATH** — Homebrew's `pkg-config` shadows the
   system one and breaks the Cray compiler wrappers' detection of
   `cray-xpmem`. Either don't set up Homebrew shellenv, or strip it from
   PATH before configuring. The toolchain block below does this defensively.

## Just set up to run an already-built `y3_cluster_cpp`

If you only want to change pipeline configuration (`.ini` files) and *not*
modify any C++/CUDA module code, this is enough.

```bash
# top of the shared install tree
export TOP_DIR=/global/common/software/des/jesteves
export COSMOSIS_REPO_DIR=${TOP_DIR}/cosmosis
export CSL_DIR=${TOP_DIR}/cosmosis-standard-library

export INTEGRATION_TOOLS_DIR=${TOP_DIR}/y3_pipe_under
export Y3PIPE_DIR=${TOP_DIR}/y3_cluster_cpp
export Y3_CLUSTER_WORK_DIR=${Y3PIPE_DIR}/release-build
export Y3_CLUSTER_CPP_DIR=${Y3PIPE_DIR}
export COSMOSIS_STANDARD_LIBRARY=${CSL_DIR}

export CUBA_DIR=${INTEGRATION_TOOLS_DIR}/cuba
export CUBA_CPP_DIR=${INTEGRATION_TOOLS_DIR}/cubacpp
export GPU_INT_DIR=${INTEGRATION_TOOLS_DIR}/gpuintegration

# avoid oversubscribing CPU cores; not carefully tuned
export OMP_NUM_THREADS=4

# CosmoSIS setup. The active conda env is y3cl_je;
# /global/common/software/des/common/Conda_Envs/cosmosis-global is OUTDATED.
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc \
       /global/common/software/des/common/Conda_Envs/y3cl_je
```

To run a CosmoSIS pipeline, allocate a GPU node and `srun`:

```bash
salloc --nodes 1 --qos interactive --time 02:00:00 \
       --constraint gpu --gpus 4 --account=des_g
cd ${Y3PIPE_DIR}/y1_rerun
srun -n 1 cosmosis --mpi demo_sigma_mort_cuda.ini
```

If something runs in a brittle way, look at `cosmosis/setup-cosmosis-nersc`
— it carries Perlmutter-specific environment hacks that may need tweaking.

## But I need to checkout and compile `y3_cluster_cpp`

This is the path most users follow. You'll work in your own copy of the
repo and rebuild the `.so` modules.

### 1. Allocate a GPU node (for tests)

```bash
salloc --nodes 1 --qos interactive --time 02:00:00 \
       --constraint gpu --gpus 4 --account=des_g
```

`cmake` and `ninja` themselves will run fine on a login node, but `ctest`
needs an actual GPU.

### 2. Set up env

Do the export block from the previous section, then source
`setup-cosmosis-nersc`:

```bash
source ${COSMOSIS_REPO_DIR}/setup-cosmosis-nersc \
       /global/common/software/des/common/Conda_Envs/y3cl_je
```

Move into your own working copy:

```bash
export MY_TOP_DIR=/global/common/software/des/$(id -un)   # or any writable dir, e.g. point to /pscratch/... 
mkdir -p ${MY_TOP_DIR}
cd ${MY_TOP_DIR}
```

### 3. Clone

Marc's repository moved from Bitbucket to GitHub. Use the GitHub URL:

```bash
git clone https://github.com/marcpaterno/y3_cluster_cpp.git
# or via SSH if your key is on GitHub:
# git clone git@github.com:marcpaterno/y3_cluster_cpp.git
```

Already have a clone? Just `git pull` inside `${Y3PIPE_DIR}`.

> The source tree **must live on a writable filesystem readable from compute
> nodes** — `/global/common/software/...` works (read+write from login,
> read-only from compute, which is fine after the build). For a fresh build
> you can also use `/pscratch/sd/<u>/<user>/...`.

### 4. Toolchain swap (do NOT take module defaults)

```bash
module swap cudatoolkit/12.9 cudatoolkit/12.2
module swap gcc-native/13.2 gcc-native/12.3
export PATH=$(echo $PATH | tr ':' '\n' | grep -v homebrew | tr '\n' ':' | sed 's/:$//')

# verify
nvcc --version    # expect 12.2
gcc --version     # expect 12.3
cmake --version   # expect 3.31.x (shared conda env was bumped to 3.31.10 on 2026-05-14)
which pkg-config  # MUST be /usr/bin/pkg-config (not ~/homebrew/bin/...)
```

**Why these specific versions?**

| Default on Perlmutter | What goes wrong | Fix |
|---|---|---|
| `cudatoolkit/12.9` | `libnvToolsExt.so` was removed → `NVTX_LIBRARY NOTFOUND` for every `gt_*_gpu` and `sigma_*` link target. | Swap to `cudatoolkit/12.2`. |
| `nvcc` ≥ 12.8.93 | Host-code `std::array` miscompile. See [NVIDIA forum thread](https://forums.developer.nvidia.com/t/new-host-code-bug-with-std-array-in-nvcc-12-8-93/328745). | Stay on CUDA 12.2 until NVIDIA ships a fix. |
| `gcc-native/13.2` | CUDA 12.2's nvcc rejects host compilers > GCC 12. Newer `glibc` also makes `SIGSTKSZ` non-constant, breaking the bundled Catch2. | Swap to `gcc-native/12.3` *and* apply the Catch2 patch below. |
| Homebrew `pkg-config` first on PATH | Cray wrappers can't find `cray-xpmem` → "C++ compiler is broken" during `cmake`'s try-compile. | Strip homebrew from PATH (the line above does this). |

### 5. Patch Catch2 (`externals/catch2/catch.hpp`)

The bundled Catch2 uses `SIGSTKSZ` as a compile-time constant. On modern
glibc it's a runtime expression; you must replace it. Three edits:

```cpp
// near line 5344 — add a constant before `static char altStackMem[];`
static constexpr std::size_t sigStackSize = 32768;
static char altStackMem[];

// near line 8525
sigStack.ss_size = sigStackSize;        // was: sigStack.ss_size = SIGSTKSZ;

// near line 8560
char FatalConditionHandler::altStackMem[FatalConditionHandler::sigStackSize] = {};
// was: char FatalConditionHandler::altStackMem[SIGSTKSZ] = {};
```

If you're cloning fresh and an existing patched copy is at hand, the
fastest sanity check is:
```bash
diff /path/to/working-copy/externals/catch2/catch.hpp \
     externals/catch2/catch.hpp
```
You should see exactly three diff hunks matching the changes above.

### 6. Configure + build

```bash
cd ${Y3PIPE_DIR}
mkdir -p release-build && cd release-build

cmake -DUSE_CUDA=On -DY3GCC_TARGET_ARCH=80-real \
      -DPAGANI_DIR=${GPU_INT_DIR} -DGSL_ROOT_DIR=${CONDA_PREFIX} \
      -DCMAKE_MODULE_PATH=${CUBA_CPP_DIR}/cmake/modules \
      -DCUBACPP_DIR=${CUBA_CPP_DIR} -DCUBA_DIR=${CUBA_DIR} \
      -DCMAKE_BUILD_TYPE=Release -G Ninja ${Y3_CLUSTER_CPP_DIR}

ninja
ctest -j 10
```

A successful configure ends with `-- Build files have been written to: .../release-build`.
If configure fails, **delete the `release-build/` directory and start over**
— stale CMake cache values cause confusing errors more often than not.

### 7. Expected results

- `cmake`: `-- Configuring done` then `-- Generating done`, no `NOTFOUND`
  variables.
- `ninja`: 336 / 336 targets, zero errors. (Last verified 2026-05-15 from
  master @ `6ef7b29` on a fresh clone.)
- `ctest -j 10`: 57 / 57 test executables passing in ~7 s.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Package cray-xpmem was not found` during `cmake` | Homebrew `pkg-config` shadows the system one | Strip `~/homebrew/bin` from PATH (step 4) |
| `unsupported GNU version` from nvcc | gcc-native 13+ left active | `module swap gcc-native/13.2 gcc-native/12.3` |
| `NVTX_LIBRARY NOTFOUND` at generate step | cudatoolkit/12.9 left active | `module swap cudatoolkit/12.9 cudatoolkit/12.2` |
| Tests fail with `REQUIRE( out.good() )` or "file not found" under `data/` | `Y3_CLUSTER_CPP_DIR` doesn't point at the source root (e.g. at `/pscratch/...`)) and `make_ofstream()` cannot create output files | run `export Y3_CLUSTER_CPP_DIR=${Y3PIPE_DIR}`, where `${Y3PIPE_DIR}` should be the path to the writable clone of the repository (e.g. under /pscratch/...). Make sure that `data/` and `validation-data/` directories exit and are writable. Then, re-run `ctest` |
| `MPI.COMM_WORLD.Get_size() == 1` under `srun` for a CosmoSIS run | conda `mpi4py` not linked to Cray MPICH | Use `cosmosis --smp=N` or a SLURM job array; see `~/.claude/CLAUDE.md` "MPI at NERSC" |
| `SIGSTKSZ` errors compiling the test binaries | Catch2 patch (step 5) wasn't applied to *this* tree | Re-apply the three patches |

### Optional: build `cuba`, `cubacpp`, or `gpuintegration` from source

You only need this if you're developing on the integration libraries
themselves; the shared `${INTEGRATION_TOOLS_DIR}` install is fine for
everyone else.

```bash
# cuba
cd ${INTEGRATION_TOOLS_DIR}
git clone https://github.com/marcpaterno/cuba.git
cd cuba
./configure
./makesharedlib.sh
mkdir include lib
mv cuba.h include/ ; mv libcuba.so lib/

# cubacpp (header-only; clone is enough unless you want to run its tests)
cd ${INTEGRATION_TOOLS_DIR}
git clone https://github.com/marcpaterno/cubacpp.git

# gpuintegration
cd ${INTEGRATION_TOOLS_DIR}
git clone https://github.com/marcpaterno/gpuintegration.git
mkdir release-build-gpuintegration
cd release-build-gpuintegration
cmake -DPAGANI_DIR=${GPU_INT_DIR} -DCMAKE_BUILD_TYPE=Release \
      -DPAGANI_TARGET_ARCH=80-real -G Ninja \
      ${INTEGRATION_TOOLS_DIR}/gpuintegration
ninja
ctest
```

## But I have to make CosmoSIS and CosmoSIS Standard Library

Take a deep breath. Then do the env exports from above.

These instructions still share the common conda environment.

```bash
cd $TOP_DIR
git clone -b y3_cluster_cpp_main https://github.com/annis/cosmosis.git
git clone -b y3_cluster_cpp_main https://github.com/annis/cosmosis-standard-library.git

cd $COSMOSIS_REPO_DIR
source setup-cosmosis-nersc /global/common/software/des/common/Conda_Envs/y3cl_je

# build
cd ${COSMOSIS_SRC_DIR}
make
cd $CSL_DIR
make

# smoke tests on a GPU node
salloc --nodes 1 --qos interactive --time 02:00:00 --constraint gpu --gpus 1 --account=des_g
srun -n 4 python -m mpi4py.bench helloworld     # validate Cray-MPI binding
cd ${COSMOSIS_STANDARD_LIBRARY}
srun -n 32 cosmosis --mpi demos/demo5.ini       # ~1 min (EMCEE / Python MPI)
srun -n 32 cosmosis --mpi demos/demo9.ini       # ~30 s  (Multinest / Fortran MPI)
```

## I want to make my own Conda environment

Only do this if you need a Python module that isn't in the shared env *and*
waiting on a request to add it isn't feasible. If you build your own conda
env you must also build your own CosmoSIS, sourcing `setup-cosmosis-nersc`
with the new env's path.

> **For PyTorch / JAX workloads, prefer `module load pytorch/2.8.0` over a
> custom conda env.** The NERSC-provided module ships torch 2.8 + CUDA 12.9
> + Python 3.12 with scipy/numpy/h5py preinstalled and is linked correctly
> against the system CUDA. See `~/.claude/CLAUDE.md` "NERSC-provided module
> envs" for the rationale.

Set the env exports, then:

```bash
conda deactivate
cd /global/common/software/des/common/Conda_Envs/

conda create --yes --prefix $NAME_OF_NEW_CONDA_ENV \
  astropy cfitsio cmake conda-tree Cython fftw fitsio gdb gh graphviz \
  gsl h5py hankel htop ipython matplotlib-base mpi4py \
  'mpich=3.3.*=external_*' ninja numpy psutil pylint python=3.9 PyYAML \
  ripgrep quarto r r-feather r-tidyverse r-tinytex scikit-learn scipy \
  tmux zeus-mcmc
```

> Do **not** update conda when it warns you a newer version is available —
> conda comes from a NERSC module, not a local install, and we can't update
> it ourselves.

The env is not yet complete; some packages have to come via `pip` to avoid
pulling in conflicting GCC-9.4 fortran/sanitizer runtimes through conda
dependencies:

```bash
conda activate $NAME_OF_NEW_CONDA_ENV

python -m pip install camb emcee future kombine mpmath sympy

# cluster_toolkit isn't on conda
mkdir -p ${INTEGRATION_TOOLS_DIR}/tmp
cd ${INTEGRATION_TOOLS_DIR}/tmp
wget https://github.com/marcpaterno/cluster_toolkit/archive/master.tar.gz
tar xf master.tar.gz
cd cluster_toolkit-master
python setup.py install
cd ${INTEGRATION_TOOLS_DIR} && rm -r tmp/
```

## Building the whole stack from scratch

If nothing exists yet, do these in order:

1. Build the conda environment (previous section).
2. Run the `pip install` block.
3. Clone + build CosmoSIS (`make` in `cosmosis`).
4. Clone + build CosmoSIS Standard Library (`make` in `cosmosis-standard-library`).
5. Clone the integration libs (`cuba`, `cubacpp`, `gpuintegration`) and
   `y3_cluster_cpp`, then follow ["But I need to checkout and compile
   `y3_cluster_cpp`"](#but-i-need-to-checkout-and-compile-y3_cluster_cpp).
