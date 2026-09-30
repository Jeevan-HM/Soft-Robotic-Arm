#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Bindings_SofaImGui" for configuration "MinSizeRel"
set_property(TARGET Bindings_SofaImGui APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(Bindings_SofaImGui PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "Python::Python"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/python3/site-packages/./SofaImGui.cpython-312-darwin.so"
  IMPORTED_SONAME_MINSIZEREL "@rpath/python3/site-packages/./SofaImGui.cpython-312-darwin.so"
  )

list(APPEND _cmake_import_check_targets Bindings_SofaImGui )
list(APPEND _cmake_import_check_files_for_Bindings_SofaImGui "${_IMPORT_PREFIX}/lib/python3/site-packages/./SofaImGui.cpython-312-darwin.so" )

# Import target "SofaImGui" for configuration "MinSizeRel"
set_property(TARGET SofaImGui APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaImGui PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "nfd::nfd"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofaImGui.26.06.00.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofaImGui.26.06.00.dylib"
  )

list(APPEND _cmake_import_check_targets SofaImGui )
list(APPEND _cmake_import_check_files_for_SofaImGui "${_IMPORT_PREFIX}/lib/libSofaImGui.26.06.00.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
