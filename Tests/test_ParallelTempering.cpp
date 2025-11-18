#include <gtest/gtest.h>
#include "ParallelTempering.h"
#include "costFunctions.h"
#include "defines.h"

using namespace FittingAlgorithms;
using namespace ParallelTempering;

double polynomicModel(double x,
                      const StringDoubleMap& fittingParams,
                      const StringDoubleMap& extraParams) {
  double a = fittingParams.at("a");
  double b = fittingParams.at("b");
  double c = fittingParams.at("c");
  return a*x*x + b*x + c;
};

// Checks that Parallel-Tempering correctly fits a second-degree polynomial
TEST(ParallelTempering, fitPolynomicFunction){
  
  std::vector<double> pointsX = {-1.3, -1.111, 0.0, 0.34, 0.76, 1.21, 2.3, 3};
  std::vector<double> pointsY(pointsX.size());

  StringDoubleMap employedParams = {{"a", 0.321}, {"b", -2.1},{"c", 1.1}};
  StringDoubleMap extraParams    = {};
  
  for(int i = 0; i<pointsX.size(); i++)
    pointsY[i] = polynomicModel(pointsX[i], employedParams, extraParams);
  
  Parameters ptParams;
  ptParams.maxIterations  = 200000;
  ptParams.temperatures   = {1e-15, 1e-12, 1e-8, 1e-4, 1e0};
  ptParams.jumpSize       = {0.00001, 0.0001, 0.001, 0.001, 0.01};
  ptParams.numStepsSwap   = 10000;
  ptParams.numStepsFinish = 200000;
  ptParams.tolerance      = 1e-20;
  ptParams.printSteps     = 100000;
    
  StringDoubleMap initialGuesses = {{"a", 10.0}, {"b", 10.0}, {"c", 6.0}};
  
  StringDoubleMap fittedParams = fit(pointsX, pointsY, polynomicModel,
                                     initialGuesses, ptParams);

  double tol = 1e-8;

  EXPECT_NEAR(fittedParams.at("a"), employedParams.at("a"), tol)
    << "Parameter 'a' not within tolerance"
    << "a: fitted=" << fittedParams.at("a")
    << ", expected=" << employedParams.at("a");
  
  EXPECT_NEAR(fittedParams.at("b"), employedParams.at("b"), tol)
    << "Parameter 'b' not within tolerance"
    << "b: fitted=" << fittedParams.at("b")
    << ", expected=" << employedParams.at("b");
  
  EXPECT_NEAR(fittedParams.at("c"), employedParams.at("c"), tol)
    << "Parameter 'c' not within tolerance"
    << "c: fitted=" << fittedParams.at("c")
    << ", expected=" << employedParams.at("c");
}
