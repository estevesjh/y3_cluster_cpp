include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

if (USE_CUDA)
  find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)

  add_library(numberCountsFull_t MODULE numberCountsFull_t.cu)
  target_link_libraries(numberCountsFull_t PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils fmt::fmt)
  target_include_directories(numberCountsFull_t PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(numberCountsFull_t PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(numberCountsFull_t PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")

  # Shear 1h:
  add_library(shear1hmissel_t MODULE Shear1hMisSel.cu)
  target_link_libraries(shear1hmissel_t PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils fmt::fmt)
  target_include_directories(shear1hmissel_t PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(shear1hmissel_t PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(shear1hmissel_t PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")

  # bSelMargGPU + shear_prj (ShearPrjEvaluator). This subdirectory was
  # never wired up on the source branch, so neither module was built.
  add_subdirectory(shear_prj_module)


endif ()
