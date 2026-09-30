#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaPython3Testing" for configuration "MinSizeRel"
set_property(TARGET SofaPython3Testing APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaPython3Testing PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofaPython3Testing.1.0.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofaPython3Testing.1.0.dylib"
  )

list(APPEND _cmake_import_check_targets SofaPython3Testing )
list(APPEND _cmake_import_check_files_for_SofaPython3Testing "${_IMPORT_PREFIX}/lib/libSofaPython3Testing.1.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
