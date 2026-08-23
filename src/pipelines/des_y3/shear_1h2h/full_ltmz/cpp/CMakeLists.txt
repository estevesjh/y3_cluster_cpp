include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# full_ltmz shear C++ backend (des_y3 Phase 2): adaptive-Cuhre triple
# integral per (bin, R), production mixture profile.
add_library(Shear1hFullLtmz MODULE Shear1hFullLtmz.cc)
target_link_libraries(Shear1hFullLtmz PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(Shear1hFullLtmz PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
