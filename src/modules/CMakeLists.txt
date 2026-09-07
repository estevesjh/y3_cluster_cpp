
#Add triaxialit
add_subdirectory(gt_card_triax_gpu)

# sigma_park_y1 is the Oct 2022 work of Jim and Johnny
# currently broken awaiting lc_lt.cuh debugging
add_subdirectory(sigma_park_y1)

# sigma_buzzard_y1 is the Nov 2022 buzzard validation dir
add_subdirectory(sigma_buzzard_y3)

# gt_mock_gpu/cpu is Sep 12 mock validation modules
add_subdirectory(gt_mock_gpu)
add_subdirectory(gt_mock_cpu)

# gt_card_gpu/cpu is Sep 16 cardinal validation modules
add_subdirectory(gt_card_gpu)
add_subdirectory(gt_card_cpu)

# gt_park_sel_gpu is April 18 cardinal validation modules
add_subdirectory(gt_park_sel_gpu)
add_subdirectory(gt_park_sel_cpu)

# sigma_mort_y1 is the primary working code of 2021 and 2022
add_subdirectory(sigma_mort_y1)
#
add_subdirectory(mass_y1)
add_subdirectory(mass_mort_y1)
add_subdirectory(y1_analysis)
add_subdirectory(sdss_analysis)
add_subdirectory(snapshotsim)
add_subdirectory(buzzard_test)
add_subdirectory(cluster_abundance_covariance)

# Closed-form richness selection N-operator (counts, M/b effective,
# Sigma/DSigma/Shear in 1h and total variants).
add_subdirectory(num_counts_sel)

# Costanzi-2026 P[X] selection-bias operators.  PAGANI variants kept as
# benchmarks; the fixed-GL evaluator (BSelMargIntegrand) is the default.
add_subdirectory(b_sel_marg_cpu)
add_subdirectory(sigma_prj_cpu)

# Brute-force triple-integral N_i[1] reference (no miscentering, no photo-z)
# used to validate the n_operator_sel family on the same datablock layout.
add_subdirectory(num_counts_full)

# Arwa Qadi's GPU projection pipeline (upstream gpu_prj_pipeline_costanzi2026).
# CUDA-only; every target inside is guarded by if (USE_CUDA).
add_subdirectory(gpu_prj_costanzi2026)

# des_y3 Phase 2 backends (src/pipelines/des_y3/README.md): new
# maintained implementations live under src/pipelines/des_y3 and are
# registered here with explicit binary dirs.
# Strategy folders are named by the number of ADAPTIVE integration
# dimensions (the cost driver): 0d = no adaptive integration at all
# (fixed-GL sums and offline tables — fast, MCMC-viable), Nd = N
# adaptive Cuhre/PAGANI dimensions (references). The maximum-dimension
# folder of each observable is always the adaptive reference; every
# lower folder is a documented dimension reduction from it. Binary
# dirs mirror the folder names.
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/number_counts/cpp/0d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_numcounts_0d_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/number_counts/cpp/3d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_numcounts_3d_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/number_counts/cuda/3d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_numcounts_3d_cuda)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/shear_1h2h/cpp/0d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear1h_0d_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/shear_1h2h/cuda/0d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear1h_0d_cuda)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/shear_1h2h/cpp/3d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear1h_3d_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/shear_1h2h/cuda/3d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear1h_3d_cuda)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/shear_projection/cpp/0d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_prj_0d_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/shear_projection/cuda/0d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_prj_0d_cuda)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/shear_projection/cpp/2d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_prj_2d_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/shear_projection/cpp/3d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_prj_3d_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/shear_projection/cuda/3d
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_prj_3d_cuda)

# Arwa Qadi's GPU P[X]/b_sel module, moved here from
# src/modules/gpu_prj_costanzi2026/shear_prj_module/ (contents unchanged).
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/systematics/selection_bias/cuda/3d
  ${CMAKE_CURRENT_BINARY_DIR}/systematics_bsel_3d_cuda)

# Many of these are historical
# add_subdirectory(DESxSPTModule)
# add_subdirectory(sigma_kappa_y1)
# add_subdirectory(compton_y_sims)
# add_subdirectory(model_sigmahm)
add_subdirectory(ExampleVector)
add_subdirectory(ExampleScalar)
add_subdirectory(ExampleOneD)
