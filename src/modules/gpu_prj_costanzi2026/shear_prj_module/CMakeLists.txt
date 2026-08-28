include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

if (USE_CUDA)
  find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)

  # bSelMargGPU.cu MOVED (file only, contents unchanged) to
  # src/pipelines/systematics/selection_bias/cuda/3d/ -- its target is
  # built from that directory's CMakeLists.txt now.

  # Projection shear
  # separate term added after shear1hmissel / numcountssel
  add_library(shear_prj MODULE ShearPrjEvaluator_t.cu)
  target_link_libraries(shear_prj PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils fmt::fmt)
  target_include_directories(shear_prj PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(shear_prj PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(shear_prj PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")



endif ()
