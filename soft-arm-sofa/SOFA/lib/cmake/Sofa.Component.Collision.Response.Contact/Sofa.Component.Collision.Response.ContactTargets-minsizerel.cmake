#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Sofa.Component.Collision.Response.Contact" for configuration "MinSizeRel"
set_property(TARGET Sofa.Component.Collision.Response.Contact APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(Sofa.Component.Collision.Response.Contact PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofa.Component.Collision.Response.Contact.26.06.00.dylib"
  IMPORTED_SONAME_MINSIZEREL "@rpath/libSofa.Component.Collision.Response.Contact.26.06.00.dylib"
  )

list(APPEND _cmake_import_check_targets Sofa.Component.Collision.Response.Contact )
list(APPEND _cmake_import_check_files_for_Sofa.Component.Collision.Response.Contact "${_IMPORT_PREFIX}/lib/libSofa.Component.Collision.Response.Contact.26.06.00.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
