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
#include "RegressionSceneList.h"

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;
#include <sofa/core/ExecParams.h>
#include <sofa/helper/logging/Messaging.h>
#include <fstream>

#include <gtest/gtest.h>

#include <string>

namespace sofa
{

template <typename T>
RegressionSceneList<T>::RegressionSceneList()
{
    const auto listType = static_cast<T*>(this)->getListType();

    char* scenesDirVar = getenv("REGRESSION_SCENES_DIR");
    if (scenesDirVar == nullptr)
    {
        msg_error(listType) << "The environment variable REGRESSION_SCENES_DIR is required but is missing or is empty.";
        return;
    }


    //Iterate through the environement variables to find substrings delimited by ':'
    size_t pos_start = 0;
    size_t pos_end;

    std::string multipleSceneDir(scenesDirVar);
    std::string tempSceneFolder;
    std::vector<std::string>  sceneFolderVector;


    //REGRESSION_SCENES_DIR
    while ((pos_end = multipleSceneDir.find('|', pos_start)) != std::string::npos)
    {
        tempSceneFolder = multipleSceneDir.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + 1;
        sceneFolderVector.push_back(tempSceneFolder);
    }
    if(multipleSceneDir.substr(pos_start,std::string::npos).size()) //get what's after the last '|' if exists
        sceneFolderVector.push_back(multipleSceneDir.substr(pos_start,std::string::npos));


    //Now gather all the regression test file in all the given directories
    for(size_t i=0; i<sceneFolderVector.size(); ++i)
    {
        if (!FileSystem::exists(FileSystem::cleanPath(sceneFolderVector[i])))
        {
            msg_error(listType) << "The environment variable REGRESSION_SCENES_DIR is invalid (its content does not exist or is incorrect); the faulty directory is " << sceneFolderVector[i];
            return;
        }
        const std::string scenesDir = std::string(FileSystem::cleanPath(sceneFolderVector[i]));

        collectScenesFromPaths(scenesDir, static_cast<T*>(this)->getListFilename());


    }
}

template <typename T>
void RegressionSceneList<T>::collectScenesFromList(const std::string& scenesDir, const std::string& listFile)
{
    // File structure
    // - # [comments]
    // - Relative path to the references
    // - reference [scene file, nb time steps, epsilon, test under mapping, test only final step]

    const std::string listDir = FileSystem::getParentDirectory(listFile);
    const std::string msgHeader = "Regression_test::" + static_cast<T*>(this)->getListType();
    msg_info(msgHeader) << "Parsing " << listFile;
    // parser le fichier -> (file,nb time steps,epsilon)
    std::ifstream iniFileStream(listFile.c_str());
    std::string referencesDir;

    //Read reference folder
    while (!iniFileStream.eof())
    {
        getline(iniFileStream, referencesDir);

        if (referencesDir.empty() || referencesDir[0] == '#')
            continue;
        else
            break;
    }

    std::string fullPathReferenceDir;
    //Check if reference dir starts with $REGRESSION_DIR
    if (referencesDir.starts_with("$REGRESSION_DIR"))
    {
        char* refDirVar = getenv("REGRESSION_DIR");
        if (refDirVar == nullptr)
        {
            msg_error(msgHeader) << "The reference path contains '$REGRESSION_DIR', and the environment variable REGRESSION_DIR is not set.";
            return;
        }
        msg_info(msgHeader)<<"Use REGRESSION_DIR as prefix";
        fullPathReferenceDir = std::string(refDirVar) + referencesDir.substr(15);
    }
    else
    {
        fullPathReferenceDir = listDir + "/" + referencesDir;
    }

    // Check if the reference folder does exist
    if (!referencesDir.empty())
    {
        msg_info(msgHeader)<<"Regression file path : "<<fullPathReferenceDir;
        if (!sofa::helper::system::FileSystem::exists(fullPathReferenceDir))
        {
            // relative reference path is wrong, check if the user set a env var instead
            // most common case: regression references are not in the same namespace (aka repository)
            char* refDirVar = getenv("REGRESSION_REFERENCES_DIR");
            if (refDirVar == nullptr)
            {
                msg_error(msgHeader) << "The reference path does not exist, and the fallback REGRESSION_REFERENCES_DIR is not set.";
                return;
            }
            else
            {
                std::string refDirFromVar(refDirVar);
                // case where REGRESSION_REFERENCES_DIR is absolute
                if (sofa::helper::system::FileSystem::isAbsolute(refDirFromVar) && sofa::helper::system::FileSystem::exists(refDirFromVar))
                {
                    fullPathReferenceDir = refDirFromVar;
                }
                else if(sofa::helper::system::FileSystem::exists(listDir + "/" + refDirFromVar))// case where REGRESSION_REFERENCES_DIR is relative
                {
                    fullPathReferenceDir = listDir + "/" + refDirFromVar;
                }
                else // all cases are not compliant, quitting
                {
                    msg_error(msgHeader) << "The reference path does not exist, and the fallback REGRESSION_REFERENCES_DIR is set but erroneous.";
                    return;
                }
            }
        }
        else
        {
            ; // OK
        }
    }
    else
    {
        msg_error(msgHeader) << "No actual content in the list files (first not commented line should be the relative path of your reference).";
        return;
    }

    msg_info(msgHeader) << "Will use the references from " << fullPathReferenceDir;

    //Read scenes
    while (!iniFileStream.eof())
    {
        std::string line;
        std::string sceneFromList;
        unsigned int steps = 100;
        double epsilon = 1e-4;
        bool testInMapping = false;
        bool dumpOnlyLastStep = false;
        int period = 0.0;

        getline(iniFileStream, line);

        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream lineStream(line);
        
        // Decompose statement
        std::vector<std::string> results(std::istream_iterator<std::string>{lineStream},
            std::istream_iterator<std::string>());
        
        if (results.size() < 3)
        {
            msg_error(msgHeader) << "Line in regression file invalid. Format should be: scene_file nbrIter threshold (testInMapping==false) (dumpOnlyLastStep==false) (period==0)";
            continue;
        }

        sceneFromList = results[0];
        steps = std::stoi(results[1]);
        epsilon = std::stod(results[2]);

        if (results.size() > 3)
            testInMapping = std::stoi(results[3]);

        if (results.size() > 4)
            dumpOnlyLastStep = std::stoi(results[4]);

        if (results.size() > 5)
            period = std::stoi(results[5]);



        std::string scene = FileSystem::append(listDir, sceneFromList);
        std::string sceneFromScenesDir(scene);
        sceneFromScenesDir.erase( sceneFromScenesDir.find(scenesDir+(scenesDir[scenesDir.size()-1] == '/' ? "" : "/")), scenesDir.size()+1 );
        std::string reference = FileSystem::append(fullPathReferenceDir, sceneFromList) + ".reference";

#ifdef WIN32
        // Minimize absolute scene path to avoid MAX_PATH problem
        if (scene.length() > MAX_PATH)
        {
            ADD_FAILURE() << scene << ": path is longer than " << MAX_PATH;
            continue;
        }
        char buffer[MAX_PATH];
        GetFullPathNameA(scene.c_str(), MAX_PATH, buffer, nullptr);
        scene = std::string(buffer);
        std::replace(scene.begin(), scene.end(), '\\', '/');
#endif // WIN32
        m_scenes.push_back( RegressionSceneData(scene, reference, steps, epsilon, testInMapping, dumpOnlyLastStep, period) );
    }
}

template <typename T>
void RegressionSceneList<T>::collectScenesFromDir(const std::string& scenesDir, const std::string& listFilename)
{
    std::vector<std::string> regressionListFiles;
    helper::system::FileSystem::findFiles(scenesDir, regressionListFiles, listFilename, 5);

    for (const std::string& regressionListFile : regressionListFiles)
    {
        if ( helper::system::FileSystem::exists(regressionListFile) && helper::system::FileSystem::isFile(regressionListFile) )
        {
            collectScenesFromList(scenesDir, regressionListFile);
        }
    }
}


template <typename T>
void RegressionSceneList<T>::collectScenesFromPaths(const std::string& scenesDir, const std::string& listFilename)
{
    collectScenesFromDir(scenesDir, listFilename); // m_sofaSrcDir should be an input to the test (not an env var)
}

} // namespace sofa
