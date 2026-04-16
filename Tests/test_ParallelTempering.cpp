#include <gtest/gtest.h>
#include "ParallelTempering.h"
#include "costFunctions.h"
#include "defines.h"
#include <complex>

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
    
  StringDoubleMap initialGuess_0 = {{"a", 10.0}, {"b", 10.0}, {"c", 6.0}};

  std::vector<StringDoubleMap> initialGuesses(ptParams.temperatures.size());
  std::fill(initialGuesses.begin(), initialGuesses.end(), initialGuess_0);
  
  StringDoubleMap fittedParams = fit<double, double>(pointsX, pointsY, polynomicModel,
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

//y = A*exp(-alpha*x) with alpha complex
std::complex<double> complexExponential(double x,
                                        const StringDoubleMap& fittingParameters,
                                        const StringDoubleMap& extraParams){
  std::complex<double> alpha(fittingParameters.at("re_alpha"),fittingParameters.at("im_alpha"));
  double A = fittingParameters.at("A");
  return A*exp(-alpha*x);
}

//y = A*exp(-alpha*x) with alpha complex
std::complex<double> complexExponentialWithPhase(double x,
                                                 const StringDoubleMap& fittingParameters,
                                                 const double phase){
  std::complex<double> alpha(fittingParameters.at("re_alpha"),fittingParameters.at("im_alpha"));
  double A = fittingParameters.at("A");
  std::complex<double> phase_im(0,-phase);
  return A*exp(-alpha*x)*exp(phase_im);
}

double squaredErrorComplex(std::complex<double> ytarget, std::complex<double> ypred){
  return std::norm(ypred - ytarget);
}

// Checks that Parallel-Tempering correctly fits a function that returns a complex number
TEST(ParallelTempering, fitComplexFunction){

  
  std::vector<double> pointsX = {0.0, 0.34, 0.76, 1.21, 2.3, 3};
  std::vector<std::complex<double>> pointsY(pointsX.size());

  double A = 0.321;
  std::complex<double> alpha(1.22, 0.23);
  
  StringDoubleMap employedParams = {{"A", A}, {"re_alpha", alpha.real()},{"im_alpha", alpha.imag()}};
  StringDoubleMap extraParams    = {};
  
  for(int i = 0; i<pointsX.size(); i++)
    pointsY[i] = complexExponential(pointsX[i], employedParams, extraParams);
  
  Parameters ptParams;
  ptParams.maxIterations  = 20000;
  ptParams.temperatures   = {1e-15, 1e-12, 1e-8, 1e-4, 1e0};
  ptParams.jumpSize       = {0.00001, 0.0001, 0.001, 0.001, 0.01};
  ptParams.numStepsSwap   = 1000;
  ptParams.numStepsFinish = 20000;
  ptParams.tolerance      = 1e-20;
  ptParams.printSteps     = 10000;
    
  StringDoubleMap initialGuess_0 = {{"A", 1.0}, {"re_alpha", 1.0}, {"im_alpha", 6.0}};

  std::vector<StringDoubleMap> initialGuesses(ptParams.temperatures.size());
  std::fill(initialGuesses.begin(), initialGuesses.end(), initialGuess_0);
  
  StringDoubleMap fittedParams = fit<double, std::complex<double>>(pointsX, pointsY, complexExponential,
                                                                   initialGuesses, ptParams, squaredErrorComplex);

  double tol = 1e-7;

  EXPECT_NEAR(fittedParams.at("A"), employedParams.at("A"), tol)
    << "Parameter 'a' not within tolerance"
    << "a: fitted=" << fittedParams.at("a")
    << ", expected=" << employedParams.at("a");
  
  EXPECT_NEAR(fittedParams.at("re_alpha"), employedParams.at("re_alpha"), tol)
    << "Parameter 're_alpha' not within tolerance"
    << "re_alpha: fitted=" << fittedParams.at("re_alpha")
    << ", expected=" << employedParams.at("re_alpha");
  
  EXPECT_NEAR(fittedParams.at("im_alpha"), employedParams.at("im_alpha"), tol)
    << "Parameter 'im_alpha' not within tolerance"
    << "im_alpha: fitted=" << fittedParams.at("im_alpha")
    << ", expected=" << employedParams.at("im_alpha");
}


// Checks that Parallel-Tempering correctly fits a function that returns a complex number
TEST(ParallelTempering, fitComplexFunctionWithPhase){

  
  std::vector<double> pointsX = {0.0, 0.34, 0.76, 1.21, 2.3, 3};
  std::vector<std::complex<double>> pointsY(pointsX.size());

  double A = 0.4321;
  std::complex<double> alpha(2.22, 0.93);
  
  StringDoubleMap employedParams = {{"A", A}, {"re_alpha", alpha.real()},{"im_alpha", alpha.imag()}};
  StringDoubleMap extraParams    = {};
  
  for(int i = 0; i<pointsX.size(); i++)
    pointsY[i] = complexExponential(pointsX[i], employedParams, extraParams);
  
  Parameters ptParams;
  ptParams.maxIterations  = 20000;
  ptParams.temperatures   = {1e-15, 1e-12, 1e-8, 1e-4, 1e0};
  ptParams.jumpSize       = {0.00001, 0.0001, 0.001, 0.001, 0.01};
  ptParams.numStepsSwap   = 1000;
  ptParams.numStepsFinish = 20000;
  ptParams.tolerance      = 1e-20;
  ptParams.printSteps     = 10000;
    
  StringDoubleMap initialGuess_0 = {{"A", 1.0}, {"re_alpha", 1.0}, {"im_alpha", 6.0}};

  std::vector<StringDoubleMap> initialGuesses(ptParams.temperatures.size());
  std::fill(initialGuesses.begin(), initialGuesses.end(), initialGuess_0);
  
  StringDoubleMap fittedParams = fit<double, std::complex<double>>(pointsX, pointsY, complexExponential,
                                                                   initialGuesses, ptParams, squaredErrorComplex);

  double tol = 1e-7;

  EXPECT_NEAR(fittedParams.at("A"), employedParams.at("A"), tol)
    << "Parameter 'a' not within tolerance"
    << "a: fitted=" << fittedParams.at("a")
    << ", expected=" << employedParams.at("a");
  
  EXPECT_NEAR(fittedParams.at("re_alpha"), employedParams.at("re_alpha"), tol)
    << "Parameter 're_alpha' not within tolerance"
    << "re_alpha: fitted=" << fittedParams.at("re_alpha")
    << ", expected=" << employedParams.at("re_alpha");
  
  EXPECT_NEAR(fittedParams.at("im_alpha"), employedParams.at("im_alpha"), tol)
    << "Parameter 'im_alpha' not within tolerance"
    << "im_alpha: fitted=" << fittedParams.at("im_alpha")
    << ", expected=" << employedParams.at("im_alpha");
}
