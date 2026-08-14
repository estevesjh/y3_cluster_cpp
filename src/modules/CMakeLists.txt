
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

# des_y3 Phase 2 backends (docs/module_reorganization_plan.md): new
# maintained implementations live under src/pipelines/des_y3 and are
# registered here with explicit binary dirs.
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/observables/shear_1h2h/radial_series/cpp
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_radial_series_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/observables/number_counts/full_ltmz/cpp
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_full_ltmz_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/observables/number_counts/full_ltmz/cuda
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_full_ltmz_cuda)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/observables/shear_1h2h/full_ltmz/cpp
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_full_ltmz_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/observables/shear_projection/fast_mass/cpp
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_prj_fast_mass_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/observables/shear_1h2h/full_ltmz/cuda
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_full_ltmz_cuda)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/observables/shear_1h2h/fast_mass/cpp
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_fast_mass_cpp)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/observables/shear_projection/full_ltmz/cuda
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_prj_full_ltmz_cuda)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/observables/shear_projection/fast_mass/cuda
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_prj_frozen_cuda)
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/src/pipelines/des_y3/observables/shear_1h2h/fast_mass/cuda
  ${CMAKE_CURRENT_BINARY_DIR}/des_y3_shear_1h2h_max_cuda)

# Many of these are historical
# add_subdirectory(DESxSPTModule)
# add_subdirectory(sigma_kappa_y1)
# add_subdirectory(compton_y_sims)
# add_subdirectory(model_sigmahm)
add_subdirectory(ExampleVector)
add_subdirectory(ExampleScalar)
add_subdirectory(ExampleOneD)
