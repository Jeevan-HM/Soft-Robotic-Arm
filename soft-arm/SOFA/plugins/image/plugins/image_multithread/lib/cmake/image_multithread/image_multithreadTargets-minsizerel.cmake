#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "image_multithread" for configuration "MinSizeRel"
set_property(TARGET image_multithread APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(image_multithread PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libimage_multithread.0.1.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libimage_multithread.0.1.dylib"
  )

list(APPEND _cmake_import_check_targets image_multithread )
list(APPEND _cmake_import_check_files_for_image_multithread "${_IMPORT_PREFIX}/lib/libimage_multithread.0.1.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
