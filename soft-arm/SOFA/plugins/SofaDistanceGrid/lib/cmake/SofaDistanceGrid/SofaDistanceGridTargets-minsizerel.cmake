#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaDistanceGrid" for configuration "MinSizeRel"
set_property(TARGET SofaDistanceGrid APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaDistanceGrid PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofaDistanceGrid.26.06.00.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofaDistanceGrid.26.06.00.dylib"
  )

list(APPEND _cmake_import_check_targets SofaDistanceGrid )
list(APPEND _cmake_import_check_files_for_SofaDistanceGrid "${_IMPORT_PREFIX}/lib/libSofaDistanceGrid.26.06.00.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
