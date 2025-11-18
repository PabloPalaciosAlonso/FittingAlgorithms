#include <gtest/gtest.h>
#include "GaussNewton.h"
#include "costFunctions.h"
#include "defines.h"

namespace FittingAlgorithms{
  namespace GaussNewton{

Eigen::MatrixXd computeJacobian(std::vector<double>& xdata,
                                std::vector<double>& ydata,
                                ModelFunction model,
                                CostFunction costFunction,
                                StringDoubleMap& paramsMap,
                                StringDoubleMap& extraParameters);
  }
}

using namespace FittingAlgorithms;
using namespace GaussNewton;

double polynomicModel(double x,
                      const StringDoubleMap& fittingParams,
                      const StringDoubleMap& extraParams) {
  double a = fittingParams.at("a");
  double b = fittingParams.at("b");
  double c = fittingParams.at("c");
  return a*x*x + b*x + c;
};

TEST(GaussNewton, JacobianPolynomicModel){

  StringDoubleMap employedParams = {{"a", 2.5}, {"b", -1.2},{"c", 0.5}};
  StringDoubleMap extraParams    = {};
  
  std::vector<double> pointsX = {-3.654, -2.342, -0.75, 0.33, 1.3, 2.44};
  std::vector<double> pointsY(pointsX.size());

  int ncols = employedParams.size();
  int nrows = pointsX.size();
  
  for(int i = 0; i<nrows; i++)
    pointsY[i] = polynomicModel(pointsX[i], employedParams, extraParams);

  std::vector<double> Jtheo(ncols*nrows);

  for (int point = 0; point<nrows; point++){
    Jtheo[ncols*point+0] = -pointsX[point]*pointsX[point];
    Jtheo[ncols*point+1] = -pointsX[point];
    Jtheo[ncols*point+2] = -1.0;
  }

  Eigen::MatrixXd Jnum = computeJacobian(pointsX, pointsY, polynomicModel,
                                         squaredError, employedParams, extraParams);
  
  for (int row = 0; row<nrows; row++){
    for (int col = 0; col<ncols; col++){
      double Jij_theo = Jtheo[col + row*ncols];
      double Jij_num  = Jnum(row, col);
      EXPECT_NEAR(Jij_theo, Jij_num, 1e-6);
    }
  }
}

// Checks that Gauss-Newton correctly fits a second-degree polynomial
TEST(GaussNewton, fitPolynomicFunction){
  
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
