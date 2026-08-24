include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# GPU adaptation of the des_y3 fast_mass max model (des_y3 Phase 2):
# host machinery verbatim, miscentred-NFW contraction as one CUDA
# kernel, reduced over z with atomicAdd.
if (USE_CUDA)
  find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)
  add_library(Shear1h2hMaxGpu MODULE Shear1h2hMaxGpu.cu)
  target_link_libraries(Shear1h2hMaxGpu PRIVATE cosmosis
                        ${NVTX_LIBRARY} utils models cuda_models
                        cuda_utils ${GSL_LIBRARIES} ${CUBA_LIBRARIES}
                        fmt::fmt)
  target_include_directories(Shear1h2hMaxGpu PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR}
    ${GSL_INCLUDE_DIRS})
  set_target_properties(Shear1h2hMaxGpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(Shear1h2hMaxGpu PRIVATE
    "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")
endif ()
