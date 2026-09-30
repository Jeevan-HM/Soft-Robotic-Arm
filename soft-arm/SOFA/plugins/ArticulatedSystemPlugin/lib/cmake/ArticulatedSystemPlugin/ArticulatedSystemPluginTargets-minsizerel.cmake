#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ArticulatedSystemPlugin" for configuration "MinSizeRel"
set_property(TARGET ArticulatedSystemPlugin APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(ArticulatedSystemPlugin PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libArticulatedSystemPlugin.1.0.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libArticulatedSystemPlugin.1.0.dylib"
  )

list(APPEND _cmake_import_check_targets ArticulatedSystemPlugin )
list(APPEND _cmake_import_check_files_for_ArticulatedSystemPlugin "${_IMPORT_PREFIX}/lib/libArticulatedSystemPlugin.1.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
