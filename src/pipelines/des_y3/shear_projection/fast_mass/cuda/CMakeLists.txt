include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# GPU adaptation of the production frozen-physics projection stage
# (des_y3 Phase 2): host machinery verbatim, DSigma_mis cache + mass
# contraction as one CUDA kernel.
if (USE_CUDA)
  find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)
  add_library(ShearPrjFrozenGpu MODULE ShearPrjFrozenGpu.cu)
  target_link_libraries(ShearPrjFrozenGpu PRIVATE cosmosis
                        ${NVTX_LIBRARY} utils models cuda_models
                        cuda_utils ${GSL_LIBRARIES} ${CUBA_LIBRARIES}
                        fmt::fmt)
  target_include_directories(ShearPrjFrozenGpu PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR}
    ${GSL_INCLUDE_DIRS})
  set_target_properties(ShearPrjFrozenGpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(ShearPrjFrozenGpu PRIVATE
    "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")
endif ()
