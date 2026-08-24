include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# Fixed-GL evaluators (one CosmoSIS module per observable).  All three
# share sp_detail::ShearPrjCore: outer theta with per-R breakpoint
# log-GL, inner (z, lnM) via ring+fg/bg GL.  Each module reads its own
# [section] block from the .ini and publishes {vals, rnd, cl} into the
# section named after itself.
#
# -ffast-math lets gcc vectorise the scalar std::sqrt/std::exp in the
# hot (zt, theta) loops; -fopenmp-simd enables #pragma omp simd
# without turning on OpenMP threading.  Scope is local so global FP
# semantics stay strict.
foreach(mod SigmaPrjEvaluator DSigmaPrjEvaluator ShearPrjEvaluator)
  add_library(${mod} MODULE ${mod}.cc)
  target_link_libraries(${mod} PRIVATE cosmosis utils models
                        ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
  target_include_directories(${mod} PRIVATE ${CMAKE_SOURCE_DIR}/src
                             ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
  target_compile_options(${mod} PRIVATE
                         -ffast-math -fopenmp-simd -march=native)
endforeach()

# The adaptive reference path (ShearPrjCuhre) moved to
# src/pipelines/des_y3/shear_projection/cpp/2d/ — it is the C++
# backend of the des_y3 full_ltmz projection strategy.

# GSL-driven variant: QAGP over z (explicit breakpoint at z=zob),
# fixed GL on (lnM, theta).  Legacy -- still on the old theta recipe
# (2-segment split at theta_R).  Kept as a z-adaptive diagnostic.
add_library(ShearPrjGsl MODULE ShearPrjGsl.cc)
target_link_libraries(ShearPrjGsl PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(ShearPrjGsl PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
target_compile_options(ShearPrjGsl PRIVATE
                       -ffast-math -fopenmp-simd -march=native)

# Frozen-physics benchmark backend (Option E of the RichnessSelection
# plan "Port frozen-physics b_sel/DeltaSigma_prj recipe into the cosmosis
# pipeline"): same rnd-hoist / a_b(z)-drift reduction as the frozen-physics
# note, but the (theta, lnM) assembly is a continuous 2-D cubacpp::Cuhre
# integral over Interp1D-tabulated w_rnd(lnM)/w_cl(lnM)/Psi(theta), instead
# of an explicit fixed grid. Diagnostic/benchmark backend, DeltaSigma-only.
add_library(ShearPrjFrozenCuhre MODULE ShearPrjFrozenCuhre.cc)
target_link_libraries(ShearPrjFrozenCuhre PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(ShearPrjFrozenCuhre PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
target_compile_options(ShearPrjFrozenCuhre PRIVATE
                       -ffast-math -fopenmp-simd -march=native)

# Option C of the same plan: same frozen-physics reduction as Option E, but
# the (theta, lnM) assembly is an explicit fixed grid + dot product (no
# Cuhre/cubacpp dependency at all) -- mirrors production ShearPrjEvaluator's
# own caching structure. Writes dsigma_prj_frozen_physics / shear_prj_frozen_physics.
add_library(ShearPrjFrozenPhysics MODULE ShearPrjFrozenPhysics.cc)
target_link_libraries(ShearPrjFrozenPhysics PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(ShearPrjFrozenPhysics PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
target_compile_options(ShearPrjFrozenPhysics PRIVATE
                       -ffast-math -fopenmp-simd -march=native)
