include_directories(${COSMOSIS_BASE_DIR} ${GSL_INCLUDE_DIRS} ${VEGAS_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

add_library(testmain SHARED main.cc)
target_include_directories(
  testmain PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/externals
  )
add_library(mpi_testmain SHARED mpi_main.cc)
target_include_directories(
  mpi_testmain PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/externals
  )
target_link_libraries(mpi_testmain PRIVATE MPI::MPI_CXX)

# Use of CUDA is optional

if (USE_CUDA)
  add_executable(cuda_interp_1d_test cuda_interp_1d.test.cu)
  target_link_libraries(cuda_interp_1d_test PRIVATE cuda_utils utils testmain)
  target_include_directories(cuda_interp_1d_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${PAGANI_DIR})
  set_target_properties(cuda_interp_1d_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  add_test(cuda_interp_1d_test cuda_interp_1d_test)

  add_executable(cuda_interp_2d_test cuda_interp_2d.test.cu)
  target_link_libraries(cuda_interp_2d_test PRIVATE cuda_utils utils testmain)
  target_include_directories(cuda_interp_2d_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${PAGANI_DIR})
  set_target_properties(cuda_interp_2d_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  add_test(cuda_interp_2d_test cuda_interp_2d_test)

  add_executable(nfw_sigma_mis_test nfw_sigma_mis.test.cu)
  target_link_libraries(nfw_sigma_mis_test PRIVATE cuda_utils utils testmain fmt::fmt)
  target_include_directories(nfw_sigma_mis_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${PAGANI_DIR})
  set_target_properties(nfw_sigma_mis_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  add_test(nfw_sigma_mis_test nfw_sigma_mis_test)

  add_executable(nfw_dsigma_mis_test nfw_dsigma_mis.test.cu)
  target_link_libraries(nfw_dsigma_mis_test PRIVATE cuda_utils utils testmain fmt::fmt)
  target_include_directories(nfw_dsigma_mis_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${PAGANI_DIR})
  set_target_properties(nfw_dsigma_mis_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  add_test(nfw_dsigma_mis_test nfw_dsigma_mis_test)

  add_executable(int_lc_lt_des_t_test int_lc_lt_des_t_test.test.cu)
  target_link_libraries(int_lc_lt_des_t_test PRIVATE cuda_utils utils testmain fmt::fmt)
  target_include_directories(int_lc_lt_des_t_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${PAGANI_DIR})
  set_target_properties(int_lc_lt_des_t_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  add_test(int_lc_lt_des_t_test int_lc_lt_des_t_test)

  add_executable(int_lc_lt_des_f_test int_lc_lt_des_f_test.test.cu)
  target_link_libraries(int_lc_lt_des_f_test PRIVATE cuda_utils utils testmain fmt::fmt)
  target_include_directories(int_lc_lt_des_f_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${PAGANI_DIR})
  set_target_properties(int_lc_lt_des_f_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH} )
  target_compile_options(int_lc_lt_des_f_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)  
  add_test(int_lc_lt_des_f_test int_lc_lt_des_f_test)
  
  add_executable(timing_sentry_cuda_test timing_sentry_test.cu)
  target_link_libraries(timing_sentry_cuda_test PRIVATE testmain)
  target_include_directories(timing_sentry_cuda_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals)
  set_target_properties(timing_sentry_cuda_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(timing_sentry_cuda_test PRIVATE "--expt-relaxed-constexpr")
  add_test(timing_sentry_cuda_test timing_sentry_cuda_test)

  # des_y3 Phase 2 CUDA backends
  find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)

  add_executable(num_counts_3d_gpu_test num_counts_3d_gpu.test.cu)
  target_link_libraries(num_counts_3d_gpu_test PRIVATE cosmosis ${NVTX_LIBRARY} cuda_models cuda_utils utils testmain fmt::fmt)
  target_include_directories(num_counts_3d_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(num_counts_3d_gpu_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(num_counts_3d_gpu_test PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")
  add_test(num_counts_3d_gpu_test num_counts_3d_gpu_test)

  add_executable(shear1h_3d_gpu_test shear1h_3d_gpu.test.cu)
  target_link_libraries(shear1h_3d_gpu_test PRIVATE cosmosis ${NVTX_LIBRARY} cuda_models cuda_utils utils testmain fmt::fmt)
  target_include_directories(shear1h_3d_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(shear1h_3d_gpu_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(shear1h_3d_gpu_test PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")
  add_test(shear1h_3d_gpu_test shear1h_3d_gpu_test)

  add_executable(shear1h2h_max_gpu_test shear1h2h_max_gpu.test.cu)
  target_link_libraries(shear1h2h_max_gpu_test PRIVATE cosmosis ${NVTX_LIBRARY} utils models cuda_models cuda_utils ${GSL_LIBRARIES} ${CUBA_LIBRARIES} testmain fmt::fmt)
  target_include_directories(shear1h2h_max_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${CUBACPP_DIR} ${PAGANI_DIR} ${GSL_INCLUDE_DIR})
  set_target_properties(shear1h2h_max_gpu_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(shear1h2h_max_gpu_test PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")
  add_test(shear1h2h_max_gpu_test shear1h2h_max_gpu_test)

  add_executable(shear_prj_frozen_gpu_test shear_prj_frozen_gpu.test.cu)
  # Default .so path = THIS build's ShearPrjFrozenGpu (the old hardcoded
  # absolute fallback silently dlopen'd a stale tree's module).
  target_compile_definitions(shear_prj_frozen_gpu_test PRIVATE
    SHEAR_PRJ_FROZEN_GPU_SO_DEFAULT="$<TARGET_FILE:ShearPrjFrozenGpu>")
  target_link_libraries(shear_prj_frozen_gpu_test PRIVATE cosmosis ${NVTX_LIBRARY} utils models cuda_models cuda_utils ${GSL_LIBRARIES} ${CUBA_LIBRARIES} testmain fmt::fmt)
  # CUBA_INCLUDE_DIR: needed now that this test directly #includes
  # shear_prj_frozen_gpu_t.cuh (for constructor-level coverage), which
  # pulls in sigma_prj_t.hh -> cubacpp/cubacpp.hh -> cuba.h. The module
  # target itself gets this via a directory-scoped include_directories()
  # in its own CMakeLists.txt; this test lives in a different directory
  # scope, so it needs it listed explicitly.
  target_include_directories(shear_prj_frozen_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${CUBACPP_DIR} ${PAGANI_DIR} ${GSL_INCLUDE_DIR} ${CUBA_INCLUDE_DIR})
  set_target_properties(shear_prj_frozen_gpu_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(shear_prj_frozen_gpu_test PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")
  add_test(shear_prj_frozen_gpu_test shear_prj_frozen_gpu_test)

  add_executable(dsigma_prj_3d_gpu_test dsigma_prj_3d_gpu.test.cu)
  target_link_libraries(dsigma_prj_3d_gpu_test PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils testmain fmt::fmt)
  target_include_directories(dsigma_prj_3d_gpu_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(dsigma_prj_3d_gpu_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(dsigma_prj_3d_gpu_test PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")
  add_test(dsigma_prj_3d_gpu_test dsigma_prj_3d_gpu_test)

endif()

add_executable(
  mpi_support_test mpi_support_test.cc
  )
target_link_libraries(
  mpi_support_test PRIVATE utils testmain
  )
target_include_directories(
  mpi_support_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(mpi_support_test mpi_support_test)

add_executable(mpi_mpi_support_test mpi_mpi_support_test.cc)
target_link_libraries(mpi_mpi_support_test PRIVATE utils mpi_testmain)
target_include_directories(mpi_mpi_support_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals)
# add_mpi_test(mpi_mpi_support_test 2)

add_executable(
  timing_sentry_test timing_sentry_test.cc
  )
target_link_libraries(
  timing_sentry_test PRIVATE testmain
  )
target_include_directories(
  timing_sentry_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(timing_sentry_test timing_sentry_test)

add_executable(
  get_grid_axes_from_datablock_test get_grid_axes_from_datablock.test.cc
  )
target_link_libraries(
  get_grid_axes_from_datablock_test PRIVATE cosmosis utils testmain
                                            ${GSL_LIBRARIES}
  )
target_include_directories(
  get_grid_axes_from_datablock_test
  PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${CMAKE_SOURCE_DIR}/externals
  )
add_test(get_grid_axes_from_datablock_test get_grid_axes_from_datablock_test)

add_executable(make_grid_splatted_test make_grid_splatted.test.cc)
target_link_libraries(
  make_grid_splatted_test PRIVATE cosmosis utils testmain ${GSL_LIBRARIES}
  )
target_include_directories(
  make_grid_splatted_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR}
                                  ${CMAKE_SOURCE_DIR}/externals
  )
add_test(make_grid_splatted_test make_grid_splatted_test)

add_executable(
  make_grid_cartesian_product_test make_grid_cartesian_product.test.cc
  )
target_link_libraries(
  make_grid_cartesian_product_test PRIVATE cosmosis utils testmain
                                           ${GSL_LIBRARIES}
  )
target_include_directories(
  make_grid_cartesian_product_test
  PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${CMAKE_SOURCE_DIR}/externals
  )
add_test(make_grid_cartesian_product_test make_grid_cartesian_product_test)

add_executable(
  make_grid_points_cartesian_product_test
  make_grid_points_cartesian_product.test.cc
  )
target_link_libraries(
  make_grid_points_cartesian_product_test PRIVATE cosmosis utils testmain
                                                  ${GSL_LIBRARIES}
  )
target_include_directories(
  make_grid_points_cartesian_product_test
  PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${CMAKE_SOURCE_DIR}/externals
  )
add_test(make_grid_points_cartesian_product_test
         make_grid_points_cartesian_product_test
         )

add_executable(make_grid_points_test make_grid_points.test.cc)
target_link_libraries(
  make_grid_points_test PRIVATE cosmosis utils testmain ${GSL_LIBRARIES}
  )
target_include_directories(
  make_grid_points_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR}
                                ${CMAKE_SOURCE_DIR}/externals
  )
add_test(make_grid_points_test make_grid_points_test)

add_executable(make_integration_volume_test make_integration_volume.test.cc)
target_link_libraries(
  make_integration_volume_test PRIVATE cosmosis utils testmain ${GSL_LIBRARIES}
  )
target_include_directories(
  make_integration_volume_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR}
                                       ${CMAKE_SOURCE_DIR}/externals
  )
add_test(make_integration_volume_test make_integration_volume_test)

add_executable(integration_test integration.test.cc)
target_link_libraries(integration_test PRIVATE testmain ${CUBA_LIBRARIES})
target_include_directories(
  integration_test
  PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
          ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )
add_test(integration_test integration_test)

add_executable(transform_test transform.test.cc)
target_link_libraries(transform_test PRIVATE testmain)
target_include_directories(
  transform_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(transform_test transform_test)

add_executable(interp_1d_test interp_1d.test.cc)
target_link_libraries(
  interp_1d_test PRIVATE testmain ${GSL_LIBRARIES} utils cosmosis
  )
target_include_directories(
  interp_1d_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
                         ${GSL_INCLUDE_DIR}
  )
add_test(interp_1d_test interp_1d_test)

add_executable(interp_2d_test interp_2d.test.cc)
target_link_libraries(
  interp_2d_test PRIVATE testmain ${GSL_LIBRARIES} utils cosmosis
  )
target_include_directories(
  interp_2d_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
                         ${GSL_INCLUDE_DIR}
  )
add_test(interp_2d_test interp_2d_test)

# radial_series C++ backend (src/pipelines/des_y3, Phase 2): table
# loading, series assembly, and the NFW_DSIGMA_MIS amplitude
# cross-check.  fmt::fmt comes in through nfw_dsigma_mis.hh.
add_executable(shear1h_radial_series_test shear1h_radial_series.test.cc)
target_link_libraries(
  shear1h_radial_series_test PRIVATE testmain ${GSL_LIBRARIES} utils
                                     cosmosis fmt::fmt
  )
target_include_directories(
  shear1h_radial_series_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR}
                                     ${CMAKE_SOURCE_DIR}/externals
                                     ${GSL_INCLUDE_DIR}
  )
add_test(shear1h_radial_series_test shear1h_radial_series_test)

# fast_mass C++ backend, shear_projection (src/pipelines/des_y3, Phase 2):
# theta-grid / exclusion-angle golden values vs the Python fast_mass
# port, the NFW_DSIGMA_MIS c=4/single convention, and ShearPrjCore's
# own wall-grid construction.
add_executable(shear_prj_gl_test shear_prj_gl.test.cc)
target_link_libraries(
  shear_prj_gl_test PRIVATE testmain ${GSL_LIBRARIES}
                                   ${CUBA_LIBRARIES} utils models cosmosis
                                   fmt::fmt
  )
target_include_directories(
  shear_prj_gl_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                                   ${CMAKE_SOURCE_DIR}/externals
                                   ${CUBACPP_DIR} ${CUBA_INCLUDE_DIR}
                                   ${GSL_INCLUDE_DIR}
  )
add_test(shear_prj_gl_test shear_prj_gl_test)

# Selection-bias closure reader (src/pipelines/systematics/selection_bias/
# cpp/bsel_bins_t.hh, and its byte-identical twin src/models/bsel_bins_t.hh):
# the exact-wall-row lookup contract every projection slice depends on.
# Previously 0% in both copies.
add_executable(bsel_bins_test bsel_bins.test.cc)
target_link_libraries(
  bsel_bins_test PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  bsel_bins_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                         ${CMAKE_SOURCE_DIR}/externals
                         ${GSL_INCLUDE_DIR}
  )
add_test(bsel_bins_test bsel_bins_test)

# Costanzi-2026 selection-bias correction B_prj(R)
# (src/pipelines/systematics/costanzi_bprj/cpp/costanzi_bprj_t.hh): closed
# form at R0, power-law limits, values-file DataBlock ctor, and the golden
# values shared with costanzi_bprj.test.py.
add_executable(costanzi_bprj_test costanzi_bprj.test.cc)
target_link_libraries(
  costanzi_bprj_test PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  costanzi_bprj_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                             ${CMAKE_SOURCE_DIR}/externals
                             ${GSL_INCLUDE_DIR}
  )
add_test(costanzi_bprj_test costanzi_bprj_test)

# End-to-end driver test for the projection-shear 0d wrapper
# (shear_projection/cpp/0d/shear_prj_gl_t.hh) and the sample-level half of
# ShearPrjCore that shear_prj_gl_test cannot reach: set_sample, the
# per-slice theta/z/NFW caches, accumulate_, locate_ and evaluate, driven
# from a small but complete synthetic sample block.
add_executable(shear_prj_gl_wrapper_test shear_prj_gl_wrapper.test.cc)
target_link_libraries(
  shear_prj_gl_wrapper_test PRIVATE testmain ${GSL_LIBRARIES}
                                    ${CUBA_LIBRARIES} utils models cosmosis
                                    fmt::fmt
  )
target_include_directories(
  shear_prj_gl_wrapper_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                                    ${CMAKE_SOURCE_DIR}/externals
                                    ${CUBACPP_DIR} ${CUBA_INCLUDE_DIR}
                                    ${GSL_INCLUDE_DIR}
  )
add_test(shear_prj_gl_wrapper_test shear_prj_gl_wrapper_test)

# Load-time smoke test for the CPU pipelines modules: every .so dlopens
# and exports the setup/execute/cleanup C ABI CosmoSIS dlsyms.  The module
# .cc translation units are otherwise never loaded by any test (all 0% in
# the coverage report), so a link regression could reach a pipeline first.
# Paths come from $<TARGET_FILE:...>, so it always tests THIS build tree.
add_executable(pipelines_module_so_test pipelines_module_so.test.cc)
target_compile_definitions(
  pipelines_module_so_test PRIVATE
    PIPELINES_SO_NUMCOUNTS_SIJ_GL="$<TARGET_FILE:NumCountsSijGl>"
    PIPELINES_SO_NUMCOUNTS_3D="$<TARGET_FILE:NumCounts3d>"
    PIPELINES_SO_SHEAR1H_GL="$<TARGET_FILE:Shear1hGl>"
    PIPELINES_SO_SHEAR1H_RADIAL_SERIES="$<TARGET_FILE:Shear1hRadialSeries>"
    PIPELINES_SO_SHEAR1H2H_MAX="$<TARGET_FILE:Shear1h2hMax>"
    PIPELINES_SO_SHEAR1H_3D="$<TARGET_FILE:Shear1h3d>"
    PIPELINES_SO_SHEAR1H2H_MAX_3D="$<TARGET_FILE:Shear1h2hMax3d>"
    PIPELINES_SO_SHEAR_PRJ_GL="$<TARGET_FILE:ShearPrjGl>"
    PIPELINES_SO_SHEAR_PRJ_CUHRE="$<TARGET_FILE:ShearPrjCuhre>"
  )
add_dependencies(
  pipelines_module_so_test
    NumCountsSijGl NumCounts3d Shear1hGl Shear1hRadialSeries Shear1h2hMax
    Shear1h3d Shear1h2hMax3d ShearPrjGl ShearPrjCuhre
  )
target_link_libraries(pipelines_module_so_test PRIVATE testmain ${CMAKE_DL_LIBS})
target_include_directories(
  pipelines_module_so_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                                   ${CMAKE_SOURCE_DIR}/externals
  )
add_test(pipelines_module_so_test pipelines_module_so_test)

# Pure-Python DES Y3 model/decomposition tests.  Use the interpreter from the
# active CosmoSIS environment so numpy/scipy match the pipeline runtime.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(
  NAME des_y3_pipeline_python_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/des_y3_pipeline.test.py
  )

# sel_function.py's own kernels (previously untested -- see
# docs/known_issues/hod_normalization_defect.md). Three sub-tests are DELIBERATELY
# red at the default 1e-3 tolerance (a real HOD-normalization finding,
# not a test miscalibration); ctest will report this target as FAILED
# until that model defect is resolved or explicitly accepted.
add_test(
  NAME sel_function_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sel_function.test.py
  )

# Full-pipeline regression against the pre-refactor implementation.  This
# checks both S_stack precision and warmed-up per-sample runtime; it skips
# itself when the Git baseline is unavailable (for example in a source
# archive rather than a checkout).
add_test(
  NAME sel_function_regression_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sel_function_regression.test.py
  )

# y3_buzzard/haloModel.py (previously untested). Characterizes the
# reported second-halo-term NaN issue (z-axis degeneracy + low-radius
# NaN fraction in dSigma_hh) as reproducible tests -- see
# docs/known_issues/dsigma_hh_debug_flag.md. Skips its second-halo-term class
# gracefully if the real_pipeline_extract_output dump isn't present.
add_test(
  NAME halo_model_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/halo_model.test.py
  )

# Cross-backend/cross-strategy consistency for number counts (full_ltmz
# Python/C++/CUDA, fast_mass Python/production C++) -- see
# docs/source/pipeline_organization.md's implemented observable matrix.
# C++/CUDA full_ltmz values are hard-coded (see the test module
# docstring for provenance/regeneration); Python backends and
# production's own output are computed/read live from the checked-in
# fiducial dump.
add_test(
  NAME numcounts_cross_backend_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/numcounts_cross_backend.test.py
  )

# Cross-backend/cross-strategy consistency for one-halo miscentred shear
# (full_ltmz Python/C++/CUDA, fast_mass Python/production C++,
# radial_series Python/C++, max-model C++/CUDA identity). C++/CUDA
# values are hard-coded (see the test module docstring for provenance);
# Python backends and production's own output are computed/read live
# from the checked-in fiducial dump.
add_test(
  NAME shear1h_cross_backend_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shear1h_cross_backend.test.py
  )

# Cross-backend/cross-strategy consistency for projection shear
# (fast_mass Python/C++/CUDA-frozen). All values are hard-coded (this
# observable has no portable checked-in dump with the projection
# sections -- see the test module docstring for provenance and the
# deliberately-out-of-scope full_ltmz CUDA backend).
add_test(
  NAME shear_prj_cross_backend_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shear_prj_cross_backend.test.py
  )

# External-validation suite (issue #11): each production input checked
# against an INDEPENDENT reference, never a port of the production code.
# Pinned values generated by test/make_external_reference_pins.py;
# tolerances are measured envelopes documented in each test's docstring.
# Both skip gracefully without the prj2h fiducial dump.
add_test(
  NAME pk_external_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pk_external.test.py
  )
add_test(
  NAME xi_nl_external_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/xi_nl_external.test.py
  )
# b_sel_marg wall vs dump-fed adaptive quad-truth. DELIBERATELY red at
# the default 1e-3: measured max devs P1 2.3e-3 / I1 1.7e-2 / J 1.9e-2,
# a real finding (coarse distances grid + wall truncation), not a test
# miscalibration -- see docs/known_issues/distances_grid_resolution_defect.md.
add_test(
  NAME bsel_external_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bsel_external.test.py
  )
# dsigma_prj rnd/cl vs dump-fed adaptive quad-truth. DELIBERATELY red at
# the default 1e-3 at the innermost radius (rnd +3.0e-2, cl +2.2e-2 --
# the known wall-edge convergence question, now quantified); the
# 2h-regime outer-radius cl guard must stay green (<= 2e-3).
add_test(
  NAME shear_prj_external_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/shear_prj_external.test.py
  )
# Concentration-mass relations vs the papers (Child+18 Eq.19/Table 2,
# Duffy+08 Table 1) + the issue-#13 conc plumbing (legacy identity,
# analytic r_s/delta_c scaling, published-table wiring).
add_test(
  NAME conc_mass_relation_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/conc_mass_relation.test.py
  )
# Signed miscentered-ΔΣ "single" table contract: negative-lobe location,
# linear (no-exp) reading, rho_mult scaling, clamping, uniform-sheet
# cancellation, Wright&Brainerd centered limit. Dump-free.
add_test(
  NAME signed_mis_table_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/signed_mis_table.test.py
  )
# Costanzi-2026 B_prj(R) Python twin (pydantic dataclass + values-file
# loader); same golden values as costanzi_bprj_test. Dump-free.
add_test(
  NAME costanzi_bprj_python_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/costanzi_bprj.test.py
  )
# likelihood_cp.py max-model mode (shear_max_section) and the Costanzi-2026
# B_prj(R) option (is_b_proj_costanzi26) on a synthetic DataBlock + DV file.
# Dump-free.
add_test(
  NAME likelihood_cp_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/likelihood_cp.test.py
  )

# S_ij-tabulated fixed-GL C++ backend, number_counts (src/pipelines/des_y3,
# 0d): the reference pipeline's production counts backend.  Pins
# NumCountsSijGl::evaluate against an independently assembled W_ij(lnM)
# sum, the include_sci = false counts contract, the HMF
# ln(m (Omega_m - Omega_nu)) mass-axis convention, per-bin S_stack
# indexing, SelGlWeights's moments, and the shared Gauss-Legendre
# primitive's exactness.
add_executable(numcounts_sij_gl_test numcounts_sij_gl.test.cc)
target_link_libraries(
  numcounts_sij_gl_test PRIVATE testmain ${GSL_LIBRARIES} ${CUBA_LIBRARIES}
                                utils models cosmosis fmt::fmt
  )
target_include_directories(
  numcounts_sij_gl_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                                ${CMAKE_SOURCE_DIR}/externals
                                ${CUBACPP_DIR} ${CUBA_INCLUDE_DIR}
                                ${GSL_INCLUDE_DIR}
  )
add_test(numcounts_sij_gl_test numcounts_sij_gl_test)

# src/pipelines/cosmology/ -- the survey-agnostic physics package, which
# had no test of its own (halo_model.test.py and conc_mass_relation.test.py
# import the y3_buzzard ORIGINALS, and only one of the two homes survives
# the merge). External references only: cluster_toolkit, colossus, and the
# Child+18 / Duffy+08 / Tinker+2010 published coefficients; plus an
# equivalence leg against the y3_buzzard copies so drift fails loudly.
add_test(
  NAME cosmology_package_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cosmology_package.test.py
  )

# Identity pins for the deliberately-duplicated pipeline sources
# (shared vs systematics sel_function/sel_kernels, cosmology vs
# systematics prj_params and bsel, src/models vs systematics
# bsel_bins_t.hh). The duplication is a KEPT decision; this asserts the
# copies still agree, and characterizes the shared sys.modules cache key
# both sel_kernels shims use.
add_test(
  NAME dual_copy_identity_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/dual_copy_identity.test.py
  )

# Convergence of shared/explicit_grid_core.py -- the project's accuracy
# REFERENCE (CLAUDE.md quotes every accuracy number against it), which
# previously had no test of its own. Checks the GL primitive against
# analytic and scipy.quad answers, and the adaptive integrator's reported
# error against scipy QUADPACK. Dump-free.
add_test(
  NAME explicit_grid_core_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/explicit_grid_core.test.py
  )

add_executable(numcounts_3d_test numcounts_3d.test.cc)
target_link_libraries(
  numcounts_3d_test PRIVATE testmain utils models cosmosis
                                   ${GSL_LIBRARIES} ${CUBA_LIBRARIES}
  )
target_include_directories(
  numcounts_3d_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                                   ${CMAKE_SOURCE_DIR}/externals
                                   ${CUBACPP_DIR} ${CUBA_INCLUDE_DIR}
                                   ${COSMOSIS_BASE_DIR} ${GSL_INCLUDE_DIR}
  )
add_test(numcounts_3d_test numcounts_3d_test)

# Traditional 1h+2h max model (des_y3 Phase 2): f_mis mixture endpoints,
# dSigma_hh NaN-sanitization, and the phi_max(1h, b*2h) composition.
add_executable(shear1h2h_max_test shear1h2h_max.test.cc)
target_link_libraries(
  shear1h2h_max_test PRIVATE testmain ${GSL_LIBRARIES} utils cosmosis
                             fmt::fmt
  )
target_include_directories(
  shear1h2h_max_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                             ${CMAKE_SOURCE_DIR}/externals
                             ${CUBACPP_DIR} ${GSL_INCLUDE_DIR}
  )
add_test(shear1h2h_max_test shear1h2h_max_test)

# The SAME observable, but driving the real header class
# (shear_1h2h/cpp/0d/shear1h2h_max_t.hh) instead of the byte-for-byte
# mirror above -- that mirror predates the class becoming public in the
# header, which is why coverage reported the header at 0% next to a
# passing shear1h2h_max_test.  Adds the z-resolved reference, the
# physical-density (1+z)^2 identity, and the ini/DataBlock contracts.
add_executable(shear1h2h_max_t_test shear1h2h_max_t.test.cc)
target_link_libraries(
  shear1h2h_max_t_test PRIVATE testmain ${GSL_LIBRARIES} ${CUBA_LIBRARIES}
                               utils models cosmosis fmt::fmt
  )
target_include_directories(
  shear1h2h_max_t_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                               ${CMAKE_SOURCE_DIR}/externals
                               ${CUBACPP_DIR} ${CUBA_INCLUDE_DIR}
                               ${GSL_INCLUDE_DIR}
  )
add_test(shear1h2h_max_t_test shear1h2h_max_t_test)

# fast_mass C++ backend, shear_1h2h (des_y3 Phase 2): SelGLCore + the
# miscentred NFW mixture vs an independently-assembled reference, and
# an identity check against the production-equivalent Shear1hMisSelGL.
add_executable(shear1h_gl_test shear1h_gl.test.cc)
target_link_libraries(
  shear1h_gl_test PRIVATE testmain ${GSL_LIBRARIES}
                                 ${CUBA_LIBRARIES} utils models cosmosis
                                 fmt::fmt
  )
target_include_directories(
  shear1h_gl_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                                 ${CMAKE_SOURCE_DIR}/externals
                                 ${CUBACPP_DIR} ${CUBA_INCLUDE_DIR}
                                 ${GSL_INCLUDE_DIR}
  )
add_test(shear1h_gl_test shear1h_gl_test)

# Shared richness/photo-z kernel headers (des_y3 Phase 2, src/models):
# the analytical P(lob|ltr,z) density vs the CDF-difference kernel
# RichnessKernel_t actually uses, cross-checked via finite-difference
# calculus and direct Simpson's-rule integration -- not just golden
# values on the fully-composed module operator().
add_executable(richness_kernel_t_test richness_kernel_t.test.cc)
target_link_libraries(
  richness_kernel_t_test PRIVATE testmain utils cosmosis
  )
target_include_directories(
  richness_kernel_t_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                                 ${CMAKE_SOURCE_DIR}/externals
  )
add_test(richness_kernel_t_test richness_kernel_t_test)

# full_ltmz C++ backend, shear_1h2h (des_y3 Phase 2): operator() vs an
# independently-assembled reference over the production sub-models, plus
# mixture-affinity and monotonic-falloff structural invariants.
add_executable(shear1h_3d_test shear1h_3d.test.cc)
target_link_libraries(
  shear1h_3d_test PRIVATE testmain ${GSL_LIBRARIES}
                                 ${CUBA_LIBRARIES} utils models cosmosis
                                 fmt::fmt
  )
target_include_directories(
  shear1h_3d_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                                 ${CMAKE_SOURCE_DIR}/externals
                                 ${CUBACPP_DIR} ${CUBA_INCLUDE_DIR}
                                 ${GSL_INCLUDE_DIR}
  )
add_test(shear1h_3d_test shear1h_3d_test)

add_executable(mz_power_law_test mz_power_law.test.cc)
target_link_libraries(mz_power_law_test PRIVATE testmain)
target_include_directories(
  mz_power_law_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                            ${CMAKE_SOURCE_DIR}/externals
  )
add_test(mz_power_law_test mz_power_law_test)

add_executable(point_3d_test point_3d.test.cc)
target_link_libraries(point_3d_test PRIVATE testmain)
target_include_directories(
  point_3d_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(point_3d_test point_3d_test)

add_executable(param_space_explorer_test param_space_explorer.test.cc)
target_link_libraries(param_space_explorer_test PRIVATE testmain)
target_include_directories(
  param_space_explorer_test PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
                                    ${CMAKE_SOURCE_DIR}/externals
  )
add_test(param_space_explorer_test param_space_explorer_test)

add_executable(fpsupport_test fpsupport.test.cc)
target_link_libraries(fpsupport_test PRIVATE testmain)
target_include_directories(
  fpsupport_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(fpsupport_test fpsupport_test)

add_executable(omega_z_sdss_test omega_z_sdss.test.cc)
target_link_libraries(
  omega_z_sdss_test PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  omega_z_sdss_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                            ${CMAKE_SOURCE_DIR}/externals
  )
add_test(omega_z_sdss_test omega_z_sdss_test)

add_executable(sample_variance_test sample_variance.test.cc)
target_link_libraries(
  sample_variance_test PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES}
                               utils models cosmosis
  )
target_include_directories(
  sample_variance_test
  PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
          ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )
# add_test(sample_variance_test sample_variance_test)

# add_executable(hmf_t_test hmf_t.test.cc) target_link_libraries(hmf_t_test
# PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} utils)
# target_include_directories(hmf_t_test PRIVATE ${CMAKE_SOURCE_DIR}/src
# ${CMAKE_SOURCE_DIR}/externals) add_test(hmf_t_test hmf_t_test)

add_executable(lo_lc_t_test lo_lc_t.test.cc)
target_link_libraries(
  lo_lc_t_test PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  lo_lc_t_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(lo_lc_t_test lo_lc_t_test)

add_executable(ez_sq_test ez_sq.test.cc)
target_link_libraries(ez_sq_test PRIVATE testmain)
target_include_directories(
  ez_sq_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(ez_sq_test ez_sq_test)

add_executable(ez_test ez.test.cc)
target_link_libraries(ez_test PRIVATE testmain)
target_include_directories(
  ez_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(ez_test ez_test)

add_executable(roffset_t_test roffset_t.test.cc)
target_link_libraries(
  roffset_t_test PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  roffset_t_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(roffset_t_test roffset_t_test)

add_executable(lc_lt_t_test lc_lt_t.test.cc)
target_link_libraries(
  lc_lt_t_test PRIVATE testmain ${GSL_LIBRARIES} utils models cosmosis
                       ${CUBA_LIBRARIES}
  )
target_include_directories(
  lc_lt_t_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(lc_lt_t_test lc_lt_t_test)

add_executable(lc_lt_probability_test lc_lt_probability_unity.test.cc)
target_link_libraries(
  lc_lt_probability_test PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES}
                                 utils models cosmosis
  )
target_include_directories(
  lc_lt_probability_test
  PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )
add_test(lc_lt_probability_test lc_lt_probability_test)

add_executable(dv_do_dz_t_test dv_do_dz_t.test.cc)
target_link_libraries(
  dv_do_dz_t_test PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} utils
                          cosmosis
  )
target_include_directories(
  dv_do_dz_t_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
                          ${CUBACPP_DIR}
  )
add_test(dv_do_dz_t_test dv_do_dz_t_test)

add_executable(polynomial_test polynomial.test.cc)
target_link_libraries(
  polynomial_test PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} utils
                          cosmosis
  )
target_include_directories(
  polynomial_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
                          ${CUBACPP_DIR}
  )
add_test(polynomial_test polynomial_test)

add_executable(integer_pow_test integer_pow.test.cc)
target_link_libraries(integer_pow_test PRIVATE testmain)
target_include_directories(
  integer_pow_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/externals
  )
add_test(integer_pow_test integer_pow_test)

add_executable(lo_lc_probability_unity_test lo_lc_probability_unity.test.cc)
target_link_libraries(
  lo_lc_probability_unity_test PRIVATE testmain ${CUBA_LIBRARIES}
                                       ${GSL_LIBRARIES} utils cosmosis
  )
target_include_directories(
  lo_lc_probability_unity_test
  PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )
add_test(lo_lc_probability_unity_test lo_lc_probability_unity_test)

add_executable(mor_t2_test mor_t2.test.cc)
target_link_libraries(
  mor_t2_test PRIVATE testmain ${GSL_LIBRARIES} utils cosmosis
  )
target_include_directories(
  mor_t2_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(mor_t2_test mor_t2_test)

add_executable(mor_probability_unity_test mor_probability_unity.test.cc)
target_link_libraries(
  mor_probability_unity_test PRIVATE testmain ${CUBA_LIBRARIES}
                                     ${GSL_LIBRARIES} utils cosmosis
  )
target_include_directories(
  mor_probability_unity_test
  PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )
add_test(mor_probability_unity_test mor_probability_unity_test)

add_executable(mor2_probability_test mor2_probability_unity.test.cc)
target_link_libraries(
  mor2_probability_test PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES}
                                utils cosmosis
  )
target_include_directories(
  mor2_probability_test
  PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )
add_test(mor2_probability_test mor2_probability_test)

add_executable(roffset_probability_test roffset_probability_unity.test.cc)
target_link_libraries(
  roffset_probability_test PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES}
                                   utils cosmosis
  )
target_include_directories(
  roffset_probability_test
  PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )
add_test(roffset_probability_test roffset_probability_test)

add_executable(integrate_delta_sigma integrate_delta_sigma.cc)
target_link_libraries(
  integrate_delta_sigma PRIVATE ${CUBA_LIBRARIES} ${GSL_LIBRARIES} utils models
                                cosmosis
  )
target_include_directories(
  integrate_delta_sigma
  PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )

add_executable(sigma_crit_inv_test sigma_crit_inv.test.cc)
target_link_libraries(
  sigma_crit_inv_test PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  sigma_crit_inv_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                              ${CMAKE_SOURCE_DIR}/externals
  )
add_test(sigma_crit_inv_test sigma_crit_inv_test)

add_executable(sin_cos_analytic_integral_test sin_cos_analytic_integral.test.cc)
target_link_libraries(
  sin_cos_analytic_integral_test PRIVATE testmain ${CUBA_LIBRARIES}
                                         ${GSL_LIBRARIES} utils models cosmosis
  )
target_include_directories(
  sin_cos_analytic_integral_test
  PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
          ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )
add_test(sin_cos_analytic_integral_test sin_cos_analytic_integral_test)

add_executable(bessel_analytic_integral_test bessel_analytic_integral.test.cc)
target_link_libraries(
  bessel_analytic_integral_test PRIVATE testmain ${CUBA_LIBRARIES}
                                        ${GSL_LIBRARIES} utils models cosmosis
  )
target_include_directories(
  bessel_analytic_integral_test
  PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
          ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )
add_test(bessel_analytic_integral_test bessel_analytic_integral_test)

add_executable(
  bessel_integral_equivalence_test bessel_integral_equivalence.test.cc
  )
target_link_libraries(
  bessel_integral_equivalence_test
  PRIVATE testmain ${CUBA_LIBRARIES} ${GSL_LIBRARIES} utils models cosmosis
  )
target_include_directories(
  bessel_integral_equivalence_test
  PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
          ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )
add_test(bessel_integral_equivalence_test bessel_integral_equivalence_test)

# add_executable(simultaneous_bin_comparison_test
# simultaneous_bin_comparison.test.cc)
# target_link_libraries(simultaneous_bin_comparison_test PRIVATE testmain
# ${CUBA_LIBRARIES} ${GSL_LIBRARIES} utils models cfitsio cosmosis)
# target_include_directories(simultaneous_bin_comparison_test PRIVATE
# ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR}
# ${CUBACPP_DIR})

add_executable(integrate_lc_lt integrate_lc_lt.cc)
target_link_libraries(
  integrate_lc_lt PRIVATE ${CUBA_LIBRARIES} ${GSL_LIBRARIES} utils models
                          cosmosis
  )
target_include_directories(
  integrate_lc_lt
  PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )

add_executable(integrate_lo_lc integrate_lo_lc.cc)
target_link_libraries(
  integrate_lo_lc PRIVATE ${CUBA_LIBRARIES} ${GSL_LIBRARIES} utils cosmosis
  )
target_include_directories(
  integrate_lo_lc
  PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )

add_executable(integrate_mor integrate_mor.cc)
target_link_libraries(
  integrate_mor PRIVATE ${CUBA_LIBRARIES} ${GSL_LIBRARIES} utils cosmosis
  )
target_include_directories(
  integrate_mor
  PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )

add_executable(integrate_roffset integrate_roffset.cc)
target_link_libraries(
  integrate_roffset PRIVATE ${CUBA_LIBRARIES} ${GSL_LIBRARIES} utils cosmosis
  )
target_include_directories(
  integrate_roffset
  PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR}
  )

add_executable(pxizeta_t_test pxizeta_t.test.cc)
target_link_libraries(
  pxizeta_t_test PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  pxizeta_t_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(pxizeta_t_test pxizeta_t_test)

add_executable(omega_z_y3xspt_test omega_z_y3xspt.test.cc)
target_link_libraries(
  omega_z_y3xspt_test PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  omega_z_y3xspt_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                              ${CMAKE_SOURCE_DIR}/externals
  )
add_test(omega_z_y3xspt_test omega_z_y3xspt_test)

add_executable(int_zo_zt_des_test int_zo_zt_des_t.test.cc)
target_link_libraries(
  int_zo_zt_des_test PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  int_zo_zt_des_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                             ${CMAKE_SOURCE_DIR}/externals
  )
add_test(int_zo_zt_des_test int_zo_zt_des_test)

add_executable(zo_zt_des_test zo_zt_des_t.test.cc)
target_link_libraries(
  zo_zt_des_test PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  zo_zt_des_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(zo_zt_des_test zo_zt_des_test)

add_executable(ln_mez_power_law ln_mez_power_law.test.cc)
target_link_libraries(
  ln_mez_power_law PRIVATE testmain utils cosmosis ${GSL_LIBRARIES}
  )
target_include_directories(
  ln_mez_power_law PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/externals
  )
add_test(ln_mez_power_law ln_mez_power_law)

add_executable(mor_y3xspt_t mor_y3xspt_t.test.cc)
target_link_libraries(
  mor_y3xspt_t PRIVATE testmain utils ${GSL_LIBRARIES} cosmosis
  )
target_include_directories(
  mor_y3xspt_t PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(mor_y3xspt_t mor_y3xspt_t)

add_executable(int_pxizeta_test int_pxizeta.test.cc)
target_link_libraries(
  int_pxizeta_test PRIVATE testmain ${GSL_LIBRARIES} utils cosmosis
  )
target_include_directories(
  int_pxizeta_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/externals
  )
add_test(int_pxizeta_test int_pxizeta_test)

add_executable(sptxdes_mor_1d sptxdes_mor_1d.test.cc)
target_link_libraries(
  sptxdes_mor_1d PRIVATE testmain ${GSL_LIBRARIES} utils cosmosis
  )
target_include_directories(
  sptxdes_mor_1d PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
  )
add_test(sptxdes_mor_1d sptxdes_mor_1d)

add_executable(
  multi_dimensional_integrator_test multi_dimensional_integrator.test.cc
  )
target_link_libraries(
  multi_dimensional_integrator_test PRIVATE testmain ${GSL_LIBRARIES}
                                            ${CUBA_LIBRARIES}
  )
target_include_directories(
  multi_dimensional_integrator_test
  PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals
          ${CUBA_INCLUDE_DIR} ${CUBACPP_DIR} ${PAGANI_DIR}
  )
add_test(multi_dimensional_integrator_test multi_dimensional_integrator_test)
