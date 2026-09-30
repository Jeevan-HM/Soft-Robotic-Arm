#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Bindings_SofaGLFW" for configuration "MinSizeRel"
set_property(TARGET Bindings_SofaGLFW APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(Bindings_SofaGLFW PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "Python::Python"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/python3/site-packages/./SofaGLFW.cpython-312-darwin.so"
  IMPORTED_SONAME_MINSIZEREL "@rpath/python3/site-packages/./SofaGLFW.cpython-312-darwin.so"
  )

list(APPEND _cmake_import_check_targets Bindings_SofaGLFW )
list(APPEND _cmake_import_check_files_for_Bindings_SofaGLFW "${_IMPORT_PREFIX}/lib/python3/site-packages/./SofaGLFW.cpython-312-darwin.so" )

# Import target "SofaGLFW" for configuration "MinSizeRel"
set_property(TARGET SofaGLFW APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaGLFW PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofaGLFW.26.06.00.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofaGLFW.26.06.00.dylib"
  )

list(APPEND _cmake_import_check_targets SofaGLFW )
list(APPEND _cmake_import_check_files_for_SofaGLFW "${_IMPORT_PREFIX}/lib/libSofaGLFW.26.06.00.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
