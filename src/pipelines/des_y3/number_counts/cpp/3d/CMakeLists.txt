include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# full_ltmz C++ backend (des_y3 Phase 2): adaptive-Cuhre triple integral
# composing the immutable HMF/volume/HOD/EMG/photo-z models.
add_library(NumCounts3d MODULE NumCounts3d.cc)
target_link_libraries(NumCounts3d PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES})
target_include_directories(NumCounts3d PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
