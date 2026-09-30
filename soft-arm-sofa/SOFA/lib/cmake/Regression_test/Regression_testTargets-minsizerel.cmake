#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Regression_test" for configuration "MinSizeRel"
set_property(TARGET Regression_test APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(Regression_test PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/Regression_test-26.06.00"
  )

list(APPEND _cmake_import_check_targets Regression_test )
list(APPEND _cmake_import_check_files_for_Regression_test "${_IMPORT_PREFIX}/bin/Regression_test-26.06.00" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
