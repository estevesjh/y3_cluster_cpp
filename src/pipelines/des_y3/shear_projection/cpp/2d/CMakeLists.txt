include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# Adaptive reference path for the projection observable (moved here
# from src/modules/sigma_prj_cpu — this IS the C++ backend of the
# des_y3 full_ltmz projection strategy): same theta outer (log-GL with
# breakpoints) but inner (z, lnM) driven by cubacpp::Cuhre /
# cubacpp::Vegas.  Used for head-to-head convergence diffs against
# ShearPrjEvaluator, Python SigmaPrj, and the fully-coupled
# DSigmaPrj3dGpu diagnostic.  Writes sigma_prj_cuhre /
# dsigma_prj_cuhre / shear_prj_cuhre (sections unchanged by the move).
add_library(ShearPrjCuhre MODULE ShearPrjCuhre.cc)
target_link_libraries(ShearPrjCuhre PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(ShearPrjCuhre PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
