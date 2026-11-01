include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# 1h z-contracted fixed-GL backend (0 adaptive dims): SelGLCore
# z-contracted weight + production mixture — one GL mass sum per
# (bin, R).
add_library(Shear1hGl MODULE Shear1hGl.cc)
target_link_libraries(Shear1hGl PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(Shear1hGl PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})

# Traditional 1h+2h max model, fixed-GL (0 adaptive dims; the 2h term
# is z-dependent so the z integral stays inside the mass integral):
# z-resolved weights x max(1h, b*2h), all tables via Interp2D.
add_library(Shear1h2hMax MODULE Shear1h2hMax.cc)
target_link_libraries(Shear1h2hMax PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES} fmt::fmt)
target_include_directories(Shear1h2hMax PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
