/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <string>
#include <vector>

namespace sofa 
{

/// a struct to store all info to perform the regression test
struct RegressionSceneData
{
    RegressionSceneData(const std::string& fileScenePath, const std::string& fileRefPath, unsigned int steps, double epsilon, bool mecaInMapping, bool dumpOnlyLastStep, int stepPeriod)
        : m_fileScenePath(fileScenePath)
        , m_fileRefPath(fileRefPath)
        , m_steps(steps)
        , m_epsilon(epsilon)
        , m_mecaInMapping(mecaInMapping)
        , m_dumpOnlyLastStep(dumpOnlyLastStep)
        , m_stepPeriod(stepPeriod)
    {}

    /// Path to the file scene to test
    std::string m_fileScenePath;
    /// Path to the reference file corresponding to the scene to test
    std::string m_fileRefPath;
    /// Number of step to perform
    unsigned int m_steps;
    /// Threshold value for dof position comparison
    double m_epsilon;
    /// Option to test mechanicalObject in Node containing a Mapping (true will test them)
    bool m_mecaInMapping;
    /// Option to compare mechanicalObject dof position at each timestep
    bool m_dumpOnlyLastStep;
    /// Option to export the data at every X steps 
    double m_stepPeriod;
};


/**
    This class will parse a given list of path (project env paths) and search for list of scene files to collect
    for future regression test.
    Main method is @sa collectScenesFromPaths
    All collected data will be store inside the vector @sa m_listScenes
*/
template <typename T>
class RegressionSceneList
{
public:
    /// List of regression Data to perform @sa RegressionSceneTest_Data
    std::vector<RegressionSceneData> m_scenes;

    RegressionSceneList();

    static inline const char* s_listSuffix = ".regression-tests";

    const std::string getListType() { return  "RegressionSceneList"; }
    const std::string getListPrefix() { return  ""; }
    const std::string getListFilename() { return static_cast<T*>(this)->getListPrefix() + std::string(s_listSuffix); }


protected:
    
    /// Method called by collectScenesFromDir to search specific regression file list inside a directory
    void collectScenesFromList(const std::string& scenesDir, const std::string& listFile);

    /// Method called by @sa collectScenesFromPaths to loop on the subdirectories to find regression file list
    void collectScenesFromDir(const std::string& scenesDir, const std::string& listFilename);

    /// Main method to start the parsing of regression file list on specific Sofa src paths
    virtual void collectScenesFromPaths(const std::string& scenesDir, const std::string& listFilename);
};

} // namespace sofa
