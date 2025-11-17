#pragma once

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include "costFunctions.h"
#include "defines.h"

namespace FittingAlgorithms{
  namespace GaussNewton{
    
    struct Parameters{
      int maxIterations     = 100;
      int printSteps        = 10;
      double tolerance      = 1e-5;
      double regularization = 1e-6;
      
    };
    
    struct FitResult {
      StringDoubleMap parameters;
      StringDoubleMap errors;
    };
    
    FitResult fit(std::vector<double>& xdata_in,
                  std::vector<double>& ydata_in,
                  ModelFunction model,
                  StringDoubleMap& initialGuesses,
                  Parameters gnParams = Parameters(),
                  CostFunction costFunction = squaredError,
                  StringDoubleMap extraParameters = {});
  }
}
