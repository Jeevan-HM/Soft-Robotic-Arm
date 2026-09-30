#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaCarving" for configuration "MinSizeRel"
set_property(TARGET SofaCarving APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaCarving PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofaCarving.0.3.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofaCarving.0.3.dylib"
  )

list(APPEND _cmake_import_check_targets SofaCarving )
list(APPEND _cmake_import_check_files_for_SofaCarving "${_IMPORT_PREFIX}/lib/libSofaCarving.0.3.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
