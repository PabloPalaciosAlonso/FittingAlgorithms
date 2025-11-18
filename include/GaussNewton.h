#pragma once

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include "costFunctions.h"
#include "defines.h"

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
 * @param gnParams        Gauss–Newton solver parameters.
 * @param costFunction    Cost function (default: squared error).
 * @param extraParameters Additional model parameters that remain fixed.
 *
 * @return FitResult containing final fitted parameters and their standard errors.
 */
  
FitResult fit(std::vector<double>& xdata_in,
              std::vector<double>& ydata_in,
              ModelFunction model,
              StringDoubleMap& initialGuesses,
              Parameters gnParams = Parameters(),
              CostFunction costFunction = squaredError,
              StringDoubleMap extraParameters = {});

} // namespace GaussNewton
} // namespace FittingAlgorithms
