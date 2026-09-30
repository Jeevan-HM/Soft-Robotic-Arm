#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMatrix" for configuration "MinSizeRel"
set_property(TARGET SofaMatrix APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaMatrix PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "metis"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofaMatrix.1.0.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofaMatrix.1.0.dylib"
  )

list(APPEND _cmake_import_check_targets SofaMatrix )
list(APPEND _cmake_import_check_files_for_SofaMatrix "${_IMPORT_PREFIX}/lib/libSofaMatrix.1.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
