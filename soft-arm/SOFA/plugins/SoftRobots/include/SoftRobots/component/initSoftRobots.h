/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture                          *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                           Plugin SoftRobots    v1.0                         *
*				                                              *
* This plugin is also distributed under the GNU LGPL (Lesser General          *
* Public License) license with the same conditions than SOFA.                 *
*                                                                             *
* Contributors: Defrost team  (INRIA, University of Lille, CNRS,              *
*               Ecole Centrale de Lille)                                      *
*                                                                             *
* Contact information: https://project.inria.fr/softrobot/contact/            *
*                                                                             *
******************************************************************************/
#ifndef INITSOFTROBOTS_H
#define INITSOFTROBOTS_H


#include <sofa/config.h>

#ifdef SOFA_BUILD_SOFTROBOTS
#define SOFA_TARGET SoftRobots
#define SOFA_SOFTROBOTS_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_SOFTROBOTS_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

#define SOFTROBOTS_HAVE_SOFAPYTHON 0
#define SOFTROBOTS_HAVE_SOFAPYTHON3 1
#define SOFTROBOTS_HAVE_SOFA_GL 1
/* #undef SOFTROBOTS_USES_COMMUNICATIONCONTROLLER */
/* #undef SOFTROBOTS_USES_GAMETRAKCONTROLLER */
/* #undef SOFTROBOTS_USES_INTERACTIVECONTROL */
/* #undef SOFTROBOTS_USES_ROBOTINOCONTROLLER */
/* #undef SOFTROBOTS_USES_WITH_CAMERA */

namespace softrobots
{
    SOFA_SOFTROBOTS_API void init();
    constexpr const char* MODULE_NAME = "SoftRobots";
    constexpr const char* MODULE_VERSION = "1.0";
}

#endif // INITSOFTROBOTS_H
