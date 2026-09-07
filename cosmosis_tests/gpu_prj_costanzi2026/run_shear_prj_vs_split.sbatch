#!/usr/bin/env bash
#SBATCH --job-name=shear_prj_vs_split
#SBATCH --account=des_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev/cosmosis_tests/gpu_prj_costanzi2026/logs/%x_%j.out
#SBATCH --error=/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev/cosmosis_tests/gpu_prj_costanzi2026/logs/%x_%j.err

set -euo pipefail
echo "host=$(hostname) jobid=${SLURM_JOB_ID}"
date

# --- env activation --------------------------------------------------
# .bashrc is NOT sourced in batch jobs. Source the project's setup script
# for its shell-side effects (COSMOSIS_STANDARD_LIBRARY, PYTHONPATH, etc.),
# then repoint at this dev tree (cosmosis_init.sh defaults to a different
# checkout) and append the pinned-toolchain CUDA 12.2 runtime -- this repo's
# CUDA modules are BUILT with cudatoolkit/12.2 (see BUILDING.md /
# build-y3-cluster-cpp skill) but the default module env here stays at
# whatever cudatoolkit release cluster_toolkit's python extension needs
# (libcudart.so.13); the 12.2 lib64 path is appended, not swapped in, so
# both resolve. Do NOT do the build-time module swaps here -- those are
# for compiling only.
source /global/homes/j/jesteves/cosmosis_init.sh

export Y3_CLUSTER_CPP_DIR=/pscratch/sd/j/jesteves/github/y3_cluster_cpp_dev
export Y3_CLUSTER_WORK_DIR=${Y3_CLUSTER_CPP_DIR}/release-build
export DES_CLUSTER_NERSC_DIR=/pscratch/sd/j/jesteves/github/des-cluster-nersc
export PYTHONPATH=${Y3_CLUSTER_CPP_DIR}:${PYTHONPATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/lib64
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${Y3_CLUSTER_WORK_DIR}/src/models:${Y3_CLUSTER_WORK_DIR}/src/utils

which cosmosis
cosmosis --version
nvidia-smi -L

cd ${Y3_CLUSTER_CPP_DIR}

# --- payload -----------------------------------------------------------
# Single run, her bselmarggpu feeding BOTH integrands (her original
# single-box shear_prj AND the 3-region PAGANI split shear_prj_split),
# both on the SAME reduced 10-row wall (bin 0, z-slice [0.20,0.35], all
# 10 canonical radii) -- direct row-for-row comparison of shear_prj/vals
# vs shear_prj_split/vals, same integrand, same b_sel input, quadrature
# only difference.
cosmosis cosmosis_tests/gpu_prj_costanzi2026/shearTot_pipeline.ini \
  -p pipeline.modules="consistency camb mstar mf_tinker prj_params halo_model sigmaCritInv average_sigma_crit_inv numberCountsFull_t shear1hmissel bselmarggpu shear_prj shear_prj_split boost_factor" test.save_dir="shear_prj_vs_split_output"

echo "done jobid=${SLURM_JOB_ID} exit=$?"
date
