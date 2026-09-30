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
#ifndef SOFA_REGRESSION_TEST_H
#define SOFA_REGRESSION_TEST_H

#include "RegressionSceneList.inl"
#include <sofa/testing/BaseTest.h>

#include <sofa/simulation/Node.h>


namespace sofa 
{

/// Regression Test general mechanism:
///
/// the class @sa RegressionSceneList is used to parse folders and look for a specific file containing the list of scene to test.
/// For state tests: RegressionStateScenes.regression-tests
/// For topology tests: RegressionTopologyScenes.regression-tests
/// Each line of the list file must contain: a local path to the scene, the number of simulation steps to run, a numerical epsilon for comparison and optinally
/// if mechanicalObject inside a mapped Node need to be tested.
///
/// Files to be tested as well as parameter parsed in the regression-tests file and reference file paths are stored in @sa RegressionSceneList.m_scenes
/// as a vector of @sa RegressionSceneData strutures
///
/// For each scene to be tested a gtest is created and the method runTest will be called to really perform the test and compare to reference files.
/// See the children method explaination for test details.
///
/// Reference files are stored in the same folder hiearchy as the tested scenes.
/// The reference files are generated when running the test for the first time on a scene and must be added to the repository: sofa-framework/regression.
/// At each commit, a test runs the scenes again for the same given number of steps.
///
/// If the result of the simulation changed voluntarily, these files must be manually deleted (locally) so they can be created again (by running the test).
/// Their modifications must be pushed to the repository.
class BaseRegression_test : public sofa::testing::BaseTest, public ::testing::WithParamInterface<RegressionSceneData>
{
public:
    /// return the name of the file being tested without path nor extension
    static std::string getTestName(const ::testing::TestParamInfo<RegressionSceneData>& p);

    /// Generic method for the test that will run the scene for a given number of steps @sa RegressionSceneData.m_steps (default 100) and call
    /// runTestImpl at each step.
    void runTest(RegressionSceneData data);

    /// Method performing the specific test of non regression. To be overwritten by child class.
    virtual void runTestImpl(RegressionSceneData data, sofa::simulation::Node::SPtr root, bool createReference = false) = 0;
};


/// Specification of @sa BaseRegression_test to perform state position comparisons
class StateRegression_test : public BaseRegression_test
{
public:
    /** Specific regression test on the states.
     * At each step, the state (position/velocity) of every independent dofs is compared to values in reference files.
     * If the mean difference per node exceed a @sa RegressionSceneData._epsilon threshold this will be reported as an error.
     */
    void runTestImpl(RegressionSceneData data, sofa::simulation::Node::SPtr root, bool createReference = false);
};

/// Specification of @sa BaseRegression_test to perform topology structure comparisons
class TopologyRegression_test : public BaseRegression_test
{
public:
    /** Specific regression test on the topology. */
    void runTestImpl(RegressionSceneData data, sofa::simulation::Node::SPtr root, bool createReference = false);
};



/// Structure creating and storing, as a List for gtest, the @sa RegressionSceneData for each Sofa scene to be tested for the state positions
struct StateRegressionSceneList : public RegressionSceneList<StateRegressionSceneList>
{
    const std::string getListType() { return  "StateRegressionSceneList"; }
    const std::string getListPrefix() { return  "RegressionStateScenes"; }

};
static StateRegressionSceneList stateRegressionSceneList; // construction will fill m_scenes


/// Structure creating and storing, as a List for gtest, the @sa RegressionSceneData for each Sofa scene to be tested for the topology structure
struct TopologyRegressionSceneList : public RegressionSceneList<TopologyRegressionSceneList>
{
    const std::string getListType() { return  "TopologyRegressionSceneList"; }
    const std::string getListPrefix() { return  "RegressionTopologyScenes"; }

};
static TopologyRegressionSceneList topologyRegressionSceneList; // construction will fill m_scenes



} // namespace sofa

#endif // SOFA_REGRESSION_TEST_H
