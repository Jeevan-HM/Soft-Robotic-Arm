#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaMJEDFEM" for configuration "MinSizeRel"
set_property(TARGET SofaMJEDFEM APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaMJEDFEM PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofaMJEDFEM.22.6.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofaMJEDFEM.22.6.dylib"
  )

list(APPEND _cmake_import_check_targets SofaMJEDFEM )
list(APPEND _cmake_import_check_files_for_SofaMJEDFEM "${_IMPORT_PREFIX}/lib/libSofaMJEDFEM.22.6.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
