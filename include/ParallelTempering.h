#pragma once
#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <vector>
#include <cfloat>
#include <omp.h>
#include<random>
#include "defines.h"
#include "costFunctions.h"

namespace FittingAlgorithms{
namespace ParallelTempering {

  /**
   * @brief Parameters controlling the Parallel Tempering optimization process.
   */
  struct Parameters{
    std::vector<double> temperatures; ///< Vector of the Monte Carlo (MC) temperatures.
    std::vector<double> jumpSize;     ///< Relative step size for each parameter being fitted.
    int numStepsSwap;                 ///< Number of Metropolis-Hastings steps between attempted chain swaps.
    int numStepsFinish;               ///< Number of steps after which execution terminates if no new global minimum has been found.
    int maxIterations;                ///< Maximum number of steps.
    double tolerance = 0;             ///< Convergence threshold.
    int printSteps;                   ///< Print intermediate results every N swap iterations.
    int numThreads = omp_get_max_threads(); ///< Number of threads to use for parallel execution.
  };

  /**
   * @brief Fits the parameters of a model to data using the Parallel Tempering algorithm.
   *
   * @param xdata_in Input X values.
   * @param ydata_in Input Y values.
   * @param model Model function of the form y = f(x, params, extraParams).
   * @param initialGuesses Vector of initial parameter maps for each chain (must match params.temperatures.size()).
   * @param params Parallel Tempering parameters.
   * @param costFunction Cost function (default: squared error).
   * @param extraParameters Additional model parameters that remain fixed.
   *
   * @return StringDoubleMap Estimated model parameters.
   */
  StringDoubleMap fit(std::vector<double> &xdata_in,
                      std::vector<double> &ydata_in,
                      ModelFunction model,
                      std::vector<StringDoubleMap> &initialGuesses,
                      Parameters params = Parameters(),
                      CostFunction costFunction = squaredError,
                      StringDoubleMap extraParameters = StringDoubleMap{});

  /**
   * @overload fit
   *
   * @brief Fits the model parameters using a single initial guess.
   *
   * This overload is equivalent to the multi-chain version, but duplicates the
   * provided initial parameter map to initialize all chains.
   *
   * @param initialGuess Initial parameter map to be replicated across all chains.
   */
  StringDoubleMap fit(std::vector<double> &xdata,
                      std::vector<double> &ydata,
                      ModelFunction model,
                      StringDoubleMap &initialGuess,
                      Parameters params = Parameters(),
                      CostFunction costFunction = squaredError,
                      StringDoubleMap extraParameters = StringDoubleMap{});


} // namespace ParallelTempering
} // namespace FittingAlgorithms

