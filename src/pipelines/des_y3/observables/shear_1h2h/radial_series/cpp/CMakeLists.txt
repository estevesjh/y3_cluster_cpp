include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# radial_series C++ backend (des_y3 Phase 2).  Thin driver over the
# immutable CosmoSISScalarEvaluatorModule template; composes the
# immutable SelGLCore and reads the committed data/radial_series text
# tables.  fmt::fmt comes in through nfw_dsigma_mis.hh (shared
# CONC/RHOC conventions), same as the num_counts_sel lensing modules.
add_library(Shear1hRadialSeries MODULE Shear1hRadialSeries.cc)
target_link_libraries(Shear1hRadialSeries PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(Shear1hRadialSeries PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
