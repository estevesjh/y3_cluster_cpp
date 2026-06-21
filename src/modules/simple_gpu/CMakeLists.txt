include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

if (USE_CUDA)
  find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)

  # bSelMargGPU.cu - Unified Costanzi-2026 selection bias module
  # Computes P1, I1, J, b_eff, B_small, B_large for gamma_prj.cuh
  add_library(bSelMargGPU MODULE bSelMargGPU.cu)
  target_link_libraries(bSelMargGPU PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils)
  target_include_directories(bSelMargGPU PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(bSelMargGPU PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(bSelMargGPU PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")

endif ()
