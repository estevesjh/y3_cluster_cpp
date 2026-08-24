include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# full_ltmz projection CUDA/PAGANI backend (des_y3 Phase 2), standard
# CUDA integration-module template.
if (USE_CUDA)
  find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)
  add_library(DSigmaPrj3dGpu MODULE DSigmaPrj3dGpu.cu)
  target_link_libraries(DSigmaPrj3dGpu PRIVATE cosmosis
                        ${NVTX_LIBRARY} utils cuda_models cuda_utils
                        fmt::fmt)
  target_include_directories(DSigmaPrj3dGpu PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(DSigmaPrj3dGpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(DSigmaPrj3dGpu PRIVATE
    "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")
endif ()
