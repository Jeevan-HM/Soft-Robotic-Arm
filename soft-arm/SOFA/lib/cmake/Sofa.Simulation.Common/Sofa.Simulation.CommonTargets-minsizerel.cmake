#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Sofa.Simulation.Common" for configuration "MinSizeRel"
set_property(TARGET Sofa.Simulation.Common APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(Sofa.Simulation.Common PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "tinyxml2::tinyxml2"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofa.Simulation.Common.26.06.00.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofa.Simulation.Common.26.06.00.dylib"
  )

list(APPEND _cmake_import_check_targets Sofa.Simulation.Common )
list(APPEND _cmake_import_check_files_for_Sofa.Simulation.Common "${_IMPORT_PREFIX}/lib/libSofa.Simulation.Common.26.06.00.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
