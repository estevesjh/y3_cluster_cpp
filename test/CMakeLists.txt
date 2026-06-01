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

  # GPU Physics Tests for Number Counts
  add_executable(gpu_number_counts_physics_test gpu_number_counts_physics.test.cu)
  target_link_libraries(gpu_number_counts_physics_test PRIVATE cuda_utils cuda_models utils testmain)
  target_include_directories(gpu_number_counts_physics_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${PAGANI_DIR})
  set_target_properties(gpu_number_counts_physics_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(gpu_number_counts_physics_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  add_test(gpu_number_counts_physics_test gpu_number_counts_physics_test)

  # GPU Physics Tests for bSelMarg
  add_executable(gpu_bsel_marg_physics_test gpu_bsel_marg_physics.test.cu)
  target_link_libraries(gpu_bsel_marg_physics_test PRIVATE cuda_utils cuda_models utils testmain)
  target_include_directories(gpu_bsel_marg_physics_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${PAGANI_DIR})
  set_target_properties(gpu_bsel_marg_physics_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(gpu_bsel_marg_physics_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  add_test(gpu_bsel_marg_physics_test gpu_bsel_marg_physics_test)

  # GPU Physics Tests for ShearPrj
  add_executable(gpu_shear_prj_physics_test gpu_shear_prj_physics.test.cu)
  target_link_libraries(gpu_shear_prj_physics_test PRIVATE cuda_utils cuda_models utils testmain)
  target_include_directories(gpu_shear_prj_physics_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${PAGANI_DIR})
  set_target_properties(gpu_shear_prj_physics_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(gpu_shear_prj_physics_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  add_test(gpu_shear_prj_physics_test gpu_shear_prj_physics_test)

  # GPU Physics Tests for Cluster Counts (Predicted N)
  add_executable(gpu_cluster_counts_physics_test gpu_cluster_counts_physics.test.cu)
  target_link_libraries(gpu_cluster_counts_physics_test PRIVATE cuda_utils cuda_models utils testmain)
  target_include_directories(gpu_cluster_counts_physics_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/externals ${PAGANI_DIR})
  set_target_properties(gpu_cluster_counts_physics_test PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(gpu_cluster_counts_physics_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  add_test(gpu_cluster_counts_physics_test gpu_cluster_counts_physics_test)

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
