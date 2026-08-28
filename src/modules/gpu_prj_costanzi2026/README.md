# GPU Projection Pipeline

This directory contains the GPU implementation of the projection pipeline based on **Costanzi et al. (2026)**.

The files added for this implementation are located in:

- `y3_cluster_cpp/src/modules/gpu_prj_costanzi2026`
- `y3_cluster_cpp/cosmosis_tests/gpu_prj_costanzi2026`

The sections below summarize the purpose of each file.

---

## src/models

### `emg_des_t.cuh`

Replaces `int_lc_lt_des_t.cuh`.

This file implements the richness selection kernel analytically using the Exponentially Modified Gaussian (EMG) cumulative distribution function (CDF). Instead of numerically integrating over the observed richness (`λ_ob`), it computes the richness selection probability from the difference of two EMG CDF evaluations.


---

### `mor_shifted_poisson_t.cuh`

Implements the continuous shifted-Poisson Mass–Observable Relation (MOR).
This replaces the skewed-Gaussian MOR implemented in `mor_des_log_t.cuh` (Costanzi et al. 2019).

Used by:

- `p_operator_gpu_t.cuh`

---

### `sigma_photoz_table_t.cuh`

Replaces `sigma_photoz_des.cuh`.

Implements the photo-z uncertainty model using tabulated values of `σ(z)`.

Used by:

- `p_operator_gpu_t.cuh`

---

### `p_operator_gpu_t.cuh`

GPU implementation of the Costanzi (2026) **P[X] operator**.

This is the CUDA equivalent of `p_operator_t.hh`. It evaluates the multidimensional projection integral used to compute the optical selection bias quantities required by the shear projection model.

---

### `b_sel.cuh`

Implements the optical selection bias model.

This file contains helper functions for computing:

- richness-to-radius conversion (`Rλ`)
- angular cluster radius (`θλ`)
- sigmoid optical bias model
- large-scale bias (`b∞`)
- small-scale bias (`b0`)
- auxiliary quantities used throughout the projection calculations

---

### `nfw_dsigma_mis.cuh`IN PROGRESS

---

### `nfw_sigma_mis.cuh`IN PROGRESS

-----

## src/modules/gpu_prj_costanzi2026

### `numberCountsFull_t.cu`

GPU implementation of the full cluster number counts model.


---

### `Shear1hMisSel.cu`

Computes the one-halo shear profile, including miscentering.

TO DO:check the miscentering in `gamma_1h_nfw.cuh`

---

## `shear_prj_module/` 

Contains the component of the 2ed halo (shear_projection).

### `bSelMargGPU.cu`

Computes the optical selection bias by marginalizing over the true richness distribution (`λ_tr`).

---

### `sigma_prj_gpu_t.cuh` IN PROGRESS

Defines the CUDA integrand used to evaluate the projected surface-density contribution (`Σ_prj`).

---

### `ShearPrjEvaluator_t.cu` IN PROGRESS

Runs the multidimensional GPU integration of the projection term and stores the resulting shear tables in the CosmoSIS DataBlock.

---

## cosmosis_tests/gpu_prj_costanzi2026

### `shearTot_pipeline.ini`

CosmoSIS pipeline for testing the projection and shear modules.


---

### `values_gpu.ini`

parameter file used by the test pipeline.

---

