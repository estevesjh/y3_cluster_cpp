include_directories(${COSMOSIS_BASE_DIR} ${CUBA_INCLUDE_DIR})
link_directories(${COSMOSIS_BASE_DIR}/cosmosis/datablock)

# full_ltmz C++ backend (des_y3 Phase 2): adaptive-Cuhre triple integral
# composing the immutable HMF/volume/HOD/EMG/photo-z models.
add_library(NumCountsFullLtmz MODULE NumCountsFullLtmz.cc)
target_link_libraries(NumCountsFullLtmz PRIVATE cosmosis utils models
                      ${GSL_LIBRARIES} ${CUBA_LIBRARIES})
target_include_directories(NumCountsFullLtmz PRIVATE
  ${CMAKE_SOURCE_DIR}/src ${CUBACPP_DIR} ${GSL_INCLUDE_DIRS})
