#pragma once

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include "costFunctions.h"
#include "defines.h"
#include "GaussNewton_detail.h"

namespace FittingAlgorithms {
namespace GaussNewton {

/**
 * @brief Parameters controlling the Gauss–Newton optimization.
 */
struct Parameters {
    int maxIterations     = 100;   ///< Maximum allowed iterations
    int printSteps        = 10;    ///< Print intermediate parameters every N iterations
    double tolerance      = 1e-5;  ///< Convergence threshold
    double regularization = 1e-6;  ///< Regularization strength
};

/**
 * @brief Result of the fitting process.
 */
struct FitResult {
    StringDoubleMap parameters;  ///< Estimated model parameters
    StringDoubleMap errors;      ///< Standard errors of the estimated parameters
};

/**
 * @brief Fits the parameters of a model to data using the Gauss–Newton algorithm.
 *
 * @param xdata_in        Input X values.
 * @param ydata_in        Input Y values.
 * @param model           Model function of the form 
 *                        y = f(x, params, extraParams),
 *                        where `params` are the parameters to be fitted
 *                        and `extraParams` are fixed parameters.
 * @param initialGuesses  Initial guesses for each fitted parameter.
 * @param gnParams        Gauss–Newton parameters.
 * @param costFunction    Cost function (default: squared error).
 * @param extraParameters Additional model parameters that remain fixed.
 *
 * @return FitResult containing final fitted parameters and their standard errors.
 */
  template <class T>
  FitResult fit(std::vector<T>& xdata_in,
                std::vector<double>& ydata_in,
                ModelFunction<T> model,
                StringDoubleMap& initialGuesses,
                Parameters gnParams = Parameters(),
                CostFunction costFunction = squaredError,
                StringDoubleMap extraParameters = {}){
    
    vector fittingParams = mapToEigen(initialGuesses);
    int n                = initialGuesses.size();
    vector y_data        = Eigen::Map<const vector>(ydata_in.data(), ydata_in.size());
    StringDoubleMap fittingParamsMap = initialGuesses;
    
    vector residual;
    for (int i = 0; i < gnParams.maxIterations; ++i) {
      
      residual = computeResiduals(xdata_in, ydata_in,
                                  model, costFunction,
                                  fittingParamsMap, extraParameters);
      
      matrix pseudoJ      = computePseudoJacobian(xdata_in, ydata_in,
                                                  model, costFunction,
                                                  gnParams.regularization,
                                                  fittingParamsMap,
                                                  extraParameters);
      vector delta_params = - pseudoJ * residual;
      double currentError = (delta_params.array() / fittingParams.array()).matrix().norm()/n;
      
      fittingParams += delta_params;
      updateMapFromEigen(fittingParamsMap, fittingParams);
      
      if (i>0 and i%gnParams.printSteps == 0)
        printParameters(fittingParamsMap, i);
      
      if (currentError < gnParams.tolerance) {
        break;
      }
    }
    
    auto errors = computeStandardErrors(xdata_in, ydata_in,
                                        model, costFunction,
                                        gnParams.regularization,
                                        fittingParamsMap,
                                        extraParameters,
                                        residual);
    
    return FitResult{fittingParamsMap, errors};
  }
  
} // namespace GaussNewton
} // namespace FittingAlgorithms
