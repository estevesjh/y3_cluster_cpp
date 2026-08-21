include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# fast_mass shear C++ backend (des_y3 Phase 2): SelGLCore + production
# mixture, namespace label/section.
add_library(Shear1hFastMass MODULE Shear1hFastMass.cc)
target_link_libraries(Shear1hFastMass PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(Shear1hFastMass PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})

# Traditional 1h+2h max model (des_y3 Phase 2): z-resolved weights (the
# 2h term is z-dependent) x max(1h, b*2h), all tables via Interp2D.
add_library(Shear1h2hMax MODULE Shear1h2hMax.cc)
target_link_libraries(Shear1h2hMax PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(Shear1h2hMax PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
