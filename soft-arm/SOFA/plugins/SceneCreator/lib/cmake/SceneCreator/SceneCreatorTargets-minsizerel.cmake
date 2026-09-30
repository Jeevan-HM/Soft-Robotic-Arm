#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SceneCreator" for configuration "MinSizeRel"
set_property(TARGET SceneCreator APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SceneCreator PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSceneCreator.0.1.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSceneCreator.0.1.dylib"
  )

list(APPEND _cmake_import_check_targets SceneCreator )
list(APPEND _cmake_import_check_files_for_SceneCreator "${_IMPORT_PREFIX}/lib/libSceneCreator.0.1.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
