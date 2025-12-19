#include <gtest/gtest.h>
#include "GaussNewton.h"
#include "costFunctions.h"
#include "defines.h"

using namespace FittingAlgorithms;
using namespace GaussNewton;

double polynomicModel1D(double x,
                        const StringDoubleMap& fittingParams,
                        const StringDoubleMap& extraParams) {
  double a = fittingParams.at("a");
  double b = fittingParams.at("b");
  double c = fittingParams.at("c");
  return a*x*x + b*x + c;
}

double nonTrivialModel1D(double x,
                         const StringDoubleMap& fittingParams,
                         const StringDoubleMap& extraParams){
  
  double a = fittingParams.at("a");
  double b = fittingParams.at("b");
  double c = fittingParams.at("c");
  
  return sqrt(c)*x*exp(-a*x*x/(b*b));
}

TEST(GaussNewton, ResidualsPolynomicModel1D){
  StringDoubleMap employedParams = {{"a", 2.5}, {"b", -1.2},{"c", 0.5}};
  StringDoubleMap extraParams    = {};
  
  std::vector<double> pointsX = {-3.654, -2.342, -0.75, 0.33, 1.3, 2.44};
  std::vector<double> pointsY(pointsX.size());
  std::vector<double> resid = {-0.32, 0.77, 0.21, 1.2, -0.3, -0.111};
  
  int ncols = employedParams.size();
  int nrows = pointsX.size();
  
  for(int i = 0; i<nrows; i++){
    pointsY[i] = polynomicModel1D(pointsX[i], employedParams, extraParams)+resid[i];
  }

  vector residNum = computeResiduals<double>(pointsX, pointsY, polynomicModel1D,
                                             squaredError, employedParams, extraParams);
  
  std::vector<double> residNumSTD = vectorToSTD(residNum);
  
  for (int i = 0; i< pointsX.size(); i++){
    EXPECT_NEAR(residNumSTD[i], resid[i], 1e-6);
  }
}

TEST(GaussNewton, ResidualsNonTrivialModel1D){
  StringDoubleMap employedParams = {{"a", 2.5}, {"b", -1.2},{"c", 0.5}};
  StringDoubleMap extraParams    = {};
  
  std::vector<double> pointsX = {-3.654, -2.342, -0.75, 0.33, 1.3, 2.44};
  std::vector<double> pointsY(pointsX.size());
  std::vector<double> resid = {-0.32, 0.77, 0.21, 1.2, -0.3, -0.111};
  
  int ncols = employedParams.size();
  int nrows = pointsX.size();
  
  for(int i = 0; i<nrows; i++){
    pointsY[i] = nonTrivialModel1D(pointsX[i], employedParams, extraParams)+resid[i];
  }

  vector residNum = computeResiduals<double>(pointsX, pointsY, nonTrivialModel1D,
                                             squaredError, employedParams, extraParams);
  
  std::vector<double> residNumSTD = vectorToSTD(residNum);
  
  for (int i = 0; i< pointsX.size(); i++){
    EXPECT_NEAR(residNumSTD[i], resid[i], 1e-6);
  }
}

TEST(GaussNewton, JacobianPolynomicModel1D){

  StringDoubleMap employedParams = {{"a", 2.5}, {"b", -1.2},{"c", 0.5}};
  StringDoubleMap extraParams    = {};
  
  std::vector<double> pointsX = {-3.654, -2.342, -0.75, 0.33, 1.3, 2.44};
  std::vector<double> pointsY(pointsX.size());

  int ncols = employedParams.size();
  int nrows = pointsX.size();
  
  for(int i = 0; i<nrows; i++)
    pointsY[i] = polynomicModel1D(pointsX[i], employedParams, extraParams);

  std::vector<double> Jtheo(ncols*nrows);

  for (int point = 0; point<nrows; point++){
    Jtheo[ncols*point+0] = -pointsX[point]*pointsX[point];
    Jtheo[ncols*point+1] = -pointsX[point];
    Jtheo[ncols*point+2] = -1.0;
  }

  matrix Jnum = computeJacobian<double>(pointsX, pointsY, polynomicModel1D,
                                        squaredError, employedParams, extraParams);
  
  for (int row = 0; row<nrows; row++){
    for (int col = 0; col<ncols; col++){
      double Jij_theo = Jtheo[col + row*ncols];
      double Jij_num  = Jnum(row, col);
      EXPECT_NEAR(Jij_theo, Jij_num, 1e-6);
    }
  }
}

TEST(GaussNewton, JacobianNonTrivialModel1D){

  double a = 2.5;
  double b = 1.2;
  double c = 0.5;
  
  StringDoubleMap employedParams = {{"a", a}, {"b", b},{"c", c}};
  StringDoubleMap extraParams    = {};
  
  std::vector<double> pointsX = {-3.654, -2.342, -0.75, 0.33, 1.3, 2.44};
  std::vector<double> pointsY(pointsX.size());

  int ncols = employedParams.size();
  int nrows = pointsX.size();
  
  for(int i = 0; i<nrows; i++)
    pointsY[i] = nonTrivialModel1D(pointsX[i], employedParams, extraParams);

  std::vector<double> Jtheo(ncols*nrows);

  for (int point = 0; point<nrows; point++){
    double x  = pointsX[point];
    double x3 = x*x*x;
    double expterm = exp(-a*x*x/(b*b));
    Jtheo[ncols*point+0] = sqrt(c)*x3/(b*b)*expterm;
    Jtheo[ncols*point+1] = -2*sqrt(c)*x3*a*expterm/(b*b*b);
    Jtheo[ncols*point+2] = -x*expterm/(2*sqrt(c));
  }

  matrix Jnum = computeJacobian<double>(pointsX, pointsY, nonTrivialModel1D,
                                        squaredError, employedParams, extraParams);
  
  for (int row = 0; row<nrows; row++){
    for (int col = 0; col<ncols; col++){
      double Jij_theo = Jtheo[col + row*ncols];
      double Jij_num  = Jnum(row, col);
      EXPECT_NEAR(Jij_theo, Jij_num, 1e-6);
    }
  }
}

// TEST(GaussNewton, PseudoJacobianPolynomicModel1D){

//   StringDoubleMap employedParams = {{"a", 2.5}, {"b", -1.2},{"c", 0.5}};
//   StringDoubleMap extraParams    = {};
//   double regularization          = 1e-5;
  
//   std::vector<double> pointsX = {0, 2};
//   std::vector<double> pointsY(pointsX.size());

//   int ncols = employedParams.size();
//   int nrows = pointsX.size();
  
//   for(int i = 0; i<nrows; i++)
//     pointsY[i] = polynomicModel1D(pointsX[i], employedParams, extraParams);

//   std::vector<double> Jtheo(ncols*nrows);

//   Jtheo[0] =  0;
//   Jtheo[1] =  2./3.;
//   Jtheo[2] =  1./2.;
//   Jtheo[3] = -1./2.;
//   Jtheo[4] = -1;
//   Jtheo[5] =  0;
  
//   matrix Jnum = computePseudoJacobian<double>(pointsX, pointsY, polynomicModel1D,
//                                               squaredError, regularization,
//                                               employedParams, extraParams);
  
//   for (int row = 0; row<nrows; row++){
//     for (int col = 0; col<ncols; col++){
//       double Jij_theo = Jtheo[col + row*ncols];
//       double Jij_num  = Jnum(col, row);
//       std::cout<<row<<" "<<col<<" "<<Jij_num<<std::endl;
//       EXPECT_NEAR(Jij_theo, Jij_num, 1e-6);
//     }
//   }
// }


// Checks that Gauss-Newton correctly fits a second-degree polynomial
TEST(GaussNewton, fitPolynomicFunction){
  
  std::vector<double> pointsX = {-1.3, -1.111, 0.0, 0.34, 0.76, 1.21, 2.3, 3};
  std::vector<double> pointsY(pointsX.size());

  StringDoubleMap employedParams = {{"a", 0.321}, {"b", -2.1},{"c", 1.1}};
  StringDoubleMap extraParams    = {};
  
  for(int i = 0; i<pointsX.size(); i++)
    pointsY[i] = polynomicModel1D(pointsX[i], employedParams, extraParams);
  
  
  FittingAlgorithms::GaussNewton::Parameters gnParams;
  gnParams.maxIterations  = 10000;
  gnParams.tolerance      = 1e-10;
  gnParams.printSteps     = 50;
  gnParams.regularization = 1e-5;

  StringDoubleMap initialGuesses = {{"a", 10.0}, {"b", 10.0}, {"c", 6.0}};

  FitResult result = fit<double>(pointsX, pointsY, polynomicModel1D,
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
