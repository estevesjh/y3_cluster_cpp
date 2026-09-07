include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# Fully-coupled adaptive-Cuhre reference for the projection observable
# (des_y3 3d strategy): CPU sibling of DSigmaPrj3dGpu, same (lntheta,
# zt, lnm) integrand (corrected 3D-ball exclusion, no w_pz, R_mis via
# the projected halo's own comoving distance), cubacpp::Cuhre instead
# of PAGANI. Two modules, identical apart from the miscentering kernel.
add_library(DSigmaPrj3d MODULE DSigmaPrj3d.cc)
target_link_libraries(DSigmaPrj3d PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(DSigmaPrj3d PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})

add_library(SigmaPrj3d MODULE SigmaPrj3d.cc)
target_link_libraries(SigmaPrj3d PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(SigmaPrj3d PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
