#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "InfinyToolkit" for configuration "MinSizeRel"
set_property(TARGET InfinyToolkit APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(InfinyToolkit PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libInfinyToolkit.0.1.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libInfinyToolkit.0.1.dylib"
  )

list(APPEND _cmake_import_check_targets InfinyToolkit )
list(APPEND _cmake_import_check_files_for_InfinyToolkit "${_IMPORT_PREFIX}/lib/libInfinyToolkit.0.1.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
