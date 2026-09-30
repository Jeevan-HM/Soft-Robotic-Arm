#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SceneChecking" for configuration "MinSizeRel"
set_property(TARGET SceneChecking APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SceneChecking PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSceneChecking.26.06.00.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSceneChecking.26.06.00.dylib"
  )

list(APPEND _cmake_import_check_targets SceneChecking )
list(APPEND _cmake_import_check_files_for_SceneChecking "${_IMPORT_PREFIX}/lib/libSceneChecking.26.06.00.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
