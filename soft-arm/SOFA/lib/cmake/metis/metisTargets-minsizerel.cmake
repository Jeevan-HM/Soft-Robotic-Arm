#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "metis" for configuration "MinSizeRel"
set_property(TARGET metis APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(metis PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/metis.framework/Versions/5.1.0/metis"
  IMPORTED_SONAME_MINSIZEREL "@rpath/metis.framework/Versions/5.1.0/metis"
  )

list(APPEND _cmake_import_check_targets metis )
list(APPEND _cmake_import_check_files_for_metis "${_IMPORT_PREFIX}/lib/metis.framework/Versions/5.1.0/metis" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
