#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Sofa.GUI" for configuration "MinSizeRel"
set_property(TARGET Sofa.GUI APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(Sofa.GUI PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofa.GUI.26.06.00.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofa.GUI.26.06.00.dylib"
  )

list(APPEND _cmake_import_check_targets Sofa.GUI )
list(APPEND _cmake_import_check_files_for_Sofa.GUI "${_IMPORT_PREFIX}/lib/libSofa.GUI.26.06.00.dylib" )

# Import target "runSofa" for configuration "MinSizeRel"
set_property(TARGET runSofa APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(runSofa PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/bin/runSofa-26.06.00"
  )

list(APPEND _cmake_import_check_targets runSofa )
list(APPEND _cmake_import_check_files_for_runSofa "${_IMPORT_PREFIX}/bin/runSofa-26.06.00" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
