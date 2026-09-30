#pragma once

#include <sofa/config.h>

#define CSPARSESOLVERS_VERSION 1.0

#ifdef SOFA_BUILD_CSPARSESOLVERS
#  define SOFA_TARGET CSparseSolvers
#  define SOFA_CSPARSESOLVERS_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#  define SOFA_CSPARSESOLVERS_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

namespace csparsesolvers
{
    constexpr const char* MODULE_NAME = "CSparseSolvers";
    constexpr const char* MODULE_VERSION = "1.0";
} // namespace csparsesolvers
