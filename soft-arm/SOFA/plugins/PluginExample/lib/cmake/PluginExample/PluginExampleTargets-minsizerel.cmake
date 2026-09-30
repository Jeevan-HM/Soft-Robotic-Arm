#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "PluginExample" for configuration "MinSizeRel"
set_property(TARGET PluginExample APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(PluginExample PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libPluginExample.1.0.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libPluginExample.1.0.dylib"
  )

list(APPEND _cmake_import_check_targets PluginExample )
list(APPEND _cmake_import_check_files_for_PluginExample "${_IMPORT_PREFIX}/lib/libPluginExample.1.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
