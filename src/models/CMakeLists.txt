link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# C++ on CPU
add_library(
  models SHARED sample_variance.cc lc_lt_t.cc lc_lt_y1_t.cc int_lc_lt_des_t.cc
  )

target_link_libraries(
  models PRIVATE utils cosmosis ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt
  )

target_include_directories(
  models
  PRIVATE ${CMAKE_SOURCE_DIR}
          ${CMAKE_SOURCE_DIR}/src
          ${CMAKE_SOURCE_DIR}/externals
          ${CUBA_INCLUDE_DIR}
          ${COSMOSIS_BASE_DIR}
          ${CUBACPP_DIR}
          ${GSL_INCLUDE_DIR}
  )

if (USE_CUDA)
    # C++ on GPU
    add_library(cuda_models SHARED int_lc_lt_des_t.cu)
    # add_library(cuda_models SHARED lc_lt_t.cu)

  target_link_libraries(cuda_models PRIVATE cosmosis utils )
  set_target_properties(
    cuda_models PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES
                                                        ${Y3GCC_TARGET_ARCH}
    )
  target_include_directories(
    cuda_models PRIVATE ${CMAKE_SOURCE_DIR}/src ${COSMOSIS_BASE_DIR}
                        ${PAGANI_DIR}
    )
  target_compile_options(cuda_models PRIVATE "--expt-relaxed-constexpr" "-lineinfo")
endif ()
