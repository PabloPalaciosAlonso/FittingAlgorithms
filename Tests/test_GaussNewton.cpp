#include <gtest/gtest.h>
#include "GaussNewton.h"

using namespace FittingAlgorithms;
using namespace GaussNewton;

// Checks that Gauss-Newton correctly fits a second-degree polynomial
TEST(GaussNewton, fitPolynomicFunction){
  
  ModelFunction polynomicModel = [](double x,
                                    const StringDoubleMap& fittingParams,
                                      const StringDoubleMap& extraParams) {
    double a = fittingParams.at("a");
    double b = fittingParams.at("b");
    double c = fittingParams.at("c");
    return a*x*x + b*x + c;
  };
  
  std::vector<double> pointsX = {-1.3, -1.111, 0.0, 0.34, 0.76, 1.21, 2.3, 3};
  std::vector<double> pointsY(pointsX.size());

  StringDoubleMap employedParams = {{"a", 0.321}, {"b", -2.1},{"c", 1.1}};
  StringDoubleMap extraParams    = {};
  
  for(int i = 0; i<pointsX.size(); i++)
    pointsY[i] = polynomicModel(pointsX[i], employedParams, extraParams);
  
  
  FittingAlgorithms::GaussNewton::Parameters gnParams;
  gnParams.maxIterations  = 10000;
  gnParams.tolerance      = 1e-10;
  gnParams.printSteps     = 50;
  gnParams.regularization = 1e-5;

  StringDoubleMap initialGuesses = {{"a", 10.0}, {"b", 10.0}, {"c", 6.0}};

  FitResult result = fit(pointsX, pointsY, polynomicModel,
                         initialGuesses, gnParams);

  StringDoubleMap fittedParams = result.parameters;

  double tol = gnParams.tolerance;

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
