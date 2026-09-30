#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TestPluginB" for configuration "MinSizeRel"
set_property(TARGET TestPluginB APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(TestPluginB PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libTestPluginB.0.7.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libTestPluginB.0.7.dylib"
  )

list(APPEND _cmake_import_check_targets TestPluginB )
list(APPEND _cmake_import_check_files_for_TestPluginB "${_IMPORT_PREFIX}/lib/libTestPluginB.0.7.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
