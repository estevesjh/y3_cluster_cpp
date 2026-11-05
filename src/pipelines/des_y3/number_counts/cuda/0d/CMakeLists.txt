include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

if (USE_CUDA)
  find_library(NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH)
  add_library(NumCountsExplicitGlGpu MODULE NumCountsExplicitGlGpu.cu)
  target_link_libraries(NumCountsExplicitGlGpu PRIVATE cosmosis
                        ${NVTX_LIBRARY} utils models cuda_models
                        cuda_utils ${GSL_LIBRARIES} ${CUBA_LIBRARIES}
                        fmt::fmt)
  target_include_directories(NumCountsExplicitGlGpu PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${PAGANI_DIR}
    ${GSL_INCLUDE_DIRS})
  set_target_properties(NumCountsExplicitGlGpu PROPERTIES
    POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES ${Y3GCC_TARGET_ARCH})
  target_compile_options(NumCountsExplicitGlGpu PRIVATE
    "--expt-relaxed-constexpr" "-lineinfo" "-DCURAND")
endif ()
