#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "DiffusionSolver" for configuration "MinSizeRel"
set_property(TARGET DiffusionSolver APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(DiffusionSolver PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libDiffusionSolver.0.1.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libDiffusionSolver.0.1.dylib"
  )

list(APPEND _cmake_import_check_targets DiffusionSolver )
list(APPEND _cmake_import_check_files_for_DiffusionSolver "${_IMPORT_PREFIX}/lib/libDiffusionSolver.0.1.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
