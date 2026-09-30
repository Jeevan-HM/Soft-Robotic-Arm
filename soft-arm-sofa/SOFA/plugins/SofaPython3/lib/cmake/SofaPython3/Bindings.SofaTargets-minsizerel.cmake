#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaPython3::Bindings.Sofa.Core" for configuration "MinSizeRel"
set_property(TARGET SofaPython3::Bindings.Sofa.Core APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaPython3::Bindings.Sofa.Core PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "Python::Python"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/python3/site-packages/Sofa/Core.cpython-312-darwin.so"
  IMPORTED_SONAME_MINSIZEREL "@rpath/python3/site-packages/Sofa/Core.cpython-312-darwin.so"
  )

list(APPEND _cmake_import_check_targets SofaPython3::Bindings.Sofa.Core )
list(APPEND _cmake_import_check_files_for_SofaPython3::Bindings.Sofa.Core "${_IMPORT_PREFIX}/lib/python3/site-packages/Sofa/Core.cpython-312-darwin.so" )

# Import target "SofaPython3::Bindings.Sofa.Helper" for configuration "MinSizeRel"
set_property(TARGET SofaPython3::Bindings.Sofa.Helper APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaPython3::Bindings.Sofa.Helper PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "Python::Python"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/python3/site-packages/Sofa/Helper.cpython-312-darwin.so"
  IMPORTED_SONAME_MINSIZEREL "@rpath/python3/site-packages/Sofa/Helper.cpython-312-darwin.so"
  )

list(APPEND _cmake_import_check_targets SofaPython3::Bindings.Sofa.Helper )
list(APPEND _cmake_import_check_files_for_SofaPython3::Bindings.Sofa.Helper "${_IMPORT_PREFIX}/lib/python3/site-packages/Sofa/Helper.cpython-312-darwin.so" )

# Import target "SofaPython3::Bindings.Sofa.Simulation" for configuration "MinSizeRel"
set_property(TARGET SofaPython3::Bindings.Sofa.Simulation APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaPython3::Bindings.Sofa.Simulation PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "Python::Python"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/python3/site-packages/Sofa/Simulation.cpython-312-darwin.so"
  IMPORTED_SONAME_MINSIZEREL "@rpath/python3/site-packages/Sofa/Simulation.cpython-312-darwin.so"
  )

list(APPEND _cmake_import_check_targets SofaPython3::Bindings.Sofa.Simulation )
list(APPEND _cmake_import_check_files_for_SofaPython3::Bindings.Sofa.Simulation "${_IMPORT_PREFIX}/lib/python3/site-packages/Sofa/Simulation.cpython-312-darwin.so" )

# Import target "SofaPython3::Bindings.Sofa.Types" for configuration "MinSizeRel"
set_property(TARGET SofaPython3::Bindings.Sofa.Types APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(SofaPython3::Bindings.Sofa.Types PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "Python::Python"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/python3/site-packages/Sofa/Types.cpython-312-darwin.so"
  IMPORTED_SONAME_MINSIZEREL "@rpath/python3/site-packages/Sofa/Types.cpython-312-darwin.so"
  )

list(APPEND _cmake_import_check_targets SofaPython3::Bindings.Sofa.Types )
list(APPEND _cmake_import_check_files_for_SofaPython3::Bindings.Sofa.Types "${_IMPORT_PREFIX}/lib/python3/site-packages/Sofa/Types.cpython-312-darwin.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
