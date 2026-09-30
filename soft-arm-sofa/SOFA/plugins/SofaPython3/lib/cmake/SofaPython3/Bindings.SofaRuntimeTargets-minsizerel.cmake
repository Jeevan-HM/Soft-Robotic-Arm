#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaPython3::Bindings.SofaRuntime" for configuration "MinSizeRel"
set_property(TARGET SofaPython3::Bindings.SofaRuntime APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaPython3::Bindings.SofaRuntime PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "Python::Python"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/SofaRuntime.cpython-312-darwin.so"
  IMPORTED_SONAME_MINSIZEREL "@rpath/python3/site-packages/SofaRuntime/SofaRuntime.cpython-312-darwin.so"
  )

list(APPEND _cmake_import_check_targets SofaPython3::Bindings.SofaRuntime )
list(APPEND _cmake_import_check_files_for_SofaPython3::Bindings.SofaRuntime "${_IMPORT_PREFIX}/lib/SofaRuntime.cpython-312-darwin.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
