include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# fast_mass projection-shear C++ backend (des_y3 Phase 2): thin driver
# over the immutable exact ShearPrjCore, both observables in one pass.
add_library(ShearPrjGl MODULE ShearPrjGl.cc)
target_link_libraries(ShearPrjGl PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(ShearPrjGl PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
