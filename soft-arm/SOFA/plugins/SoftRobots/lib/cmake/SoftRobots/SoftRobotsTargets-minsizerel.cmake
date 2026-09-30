#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SoftRobots" for configuration "MinSizeRel"
set_property(TARGET SoftRobots APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SoftRobots PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSoftRobots.1.0.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSoftRobots.1.0.dylib"
  )

list(APPEND _cmake_import_check_targets SoftRobots )
list(APPEND _cmake_import_check_files_for_SoftRobots "${_IMPORT_PREFIX}/lib/libSoftRobots.1.0.dylib" )

# Import target "SoftRobotsInverseBindings" for configuration "MinSizeRel"
set_property(TARGET SoftRobotsInverseBindings APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SoftRobotsInverseBindings PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "Python::Python"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/python3/site-packages/Sofa/SoftRobotsInverse.cpython-312-darwin.so"
  IMPORTED_SONAME_MINSIZEREL "@rpath/python3/site-packages/Sofa/SoftRobotsInverse.cpython-312-darwin.so"
  )

list(APPEND _cmake_import_check_targets SoftRobotsInverseBindings )
list(APPEND _cmake_import_check_files_for_SoftRobotsInverseBindings "${_IMPORT_PREFIX}/lib/python3/site-packages/Sofa/SoftRobotsInverse.cpython-312-darwin.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
