#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TestPluginC" for configuration "MinSizeRel"
set_property(TARGET TestPluginC APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(TestPluginC PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libTestPluginC.0.7.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libTestPluginC.0.7.dylib"
  )

list(APPEND _cmake_import_check_targets TestPluginC )
list(APPEND _cmake_import_check_files_for_TestPluginC "${_IMPORT_PREFIX}/lib/libTestPluginC.0.7.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
