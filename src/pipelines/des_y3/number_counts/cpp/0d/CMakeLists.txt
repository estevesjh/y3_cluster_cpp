include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# fast_mass number-counts C++ backend (des_y3 Phase 2): the des_y3
# namespace label/section wrapping the same SelGLCore-based algorithm
# as production NumCountsSel.so ("by identity", src/pipelines/des_y3/README.md).
add_library(NumCountsSijGl MODULE NumCountsSijGl.cc)
target_link_libraries(NumCountsSijGl PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(NumCountsSijGl PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
