include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

if (USE_CUDA)
  find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)

  # sigma1hSimple.cu - 1-halo surface density without selection function
  add_library(sigma1hSimple MODULE sigma1hSimple.cu)
  target_link_libraries(sigma1hSimple PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils)
  target_include_directories(sigma1hSimple PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(sigma1hSimple PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(sigma1hSimple PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")

  # sigmaTotSimple.cu - Total (1h + b*2h) surface density without selection function
  add_library(sigmaTotSimple MODULE sigmaTotSimple.cu)
  target_link_libraries(sigmaTotSimple PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils)
  target_include_directories(sigmaTotSimple PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(sigmaTotSimple PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(sigmaTotSimple PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")

  # shearTotSimple.cu - Total tangential shear without selection function
  add_library(shearTotSimple MODULE shearTotSimple.cu)
  target_link_libraries(shearTotSimple PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils)
  target_include_directories(shearTotSimple PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(shearTotSimple PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(shearTotSimple PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")

  # numberCountsFull.cu - Full 4D integration over (lob, ltr, zt, lnM)
  # This module integrates directly without using pre-tabulated S_ij
  add_library(numberCountsFull MODULE numberCountsFull.cu)
  target_link_libraries(numberCountsFull PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils)
  target_include_directories(numberCountsFull PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(numberCountsFull PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(numberCountsFull PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")

  # bSelMargGPU.cu - Unified Costanzi-2026 (P1, I1, J) operators
  # Adapts p_operator_cuhre_t.hh for GPU with 3D integration + inner GL theta loop
  add_library(bSelMargGPU MODULE bSelMargGPU.cu)
  target_link_libraries(bSelMargGPU PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils)
  target_include_directories(bSelMargGPU PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(bSelMargGPU PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(bSelMargGPU PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")

  # ShearPrjGPU.cu - GPU shear projection (sigma_prj, dsigma_prj, shear_prj)
  # Computes 2-halo tangential shear with selection bias
  add_library(ShearPrjGPU MODULE ShearPrjGPU.cu)
  target_link_libraries(ShearPrjGPU PRIVATE cosmosis ${NVTX_LIBRARY} utils cuda_models cuda_utils)
  target_include_directories(ShearPrjGPU PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(ShearPrjGPU PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(ShearPrjGPU PRIVATE "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")

endif ()
