#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaImplicitField" for configuration "MinSizeRel"
set_property(TARGET SofaImplicitField APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaImplicitField PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "Sofa.Component.Topology.Container.Constant"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofaImplicitField.1.0.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofaImplicitField.1.0.dylib"
  )

list(APPEND _cmake_import_check_targets SofaImplicitField )
list(APPEND _cmake_import_check_files_for_SofaImplicitField "${_IMPORT_PREFIX}/lib/libSofaImplicitField.1.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
