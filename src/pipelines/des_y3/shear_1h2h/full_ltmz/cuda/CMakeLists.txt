include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# full_ltmz shear CUDA/PAGANI backend (des_y3 Phase 2).
if (USE_CUDA)
  find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)
  add_library(Shear1hFullLtmzGpu MODULE Shear1hFullLtmzGpu.cu)
  target_link_libraries(Shear1hFullLtmzGpu PRIVATE cosmosis
                        ${NVTX_LIBRARY} utils cuda_models cuda_utils
                        fmt::fmt)
  target_include_directories(Shear1hFullLtmzGpu PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR})
  set_target_properties(Shear1hFullLtmzGpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(Shear1hFullLtmzGpu PRIVATE
    "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")
endif ()
