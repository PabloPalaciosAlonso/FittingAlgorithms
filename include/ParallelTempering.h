#pragma once
#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <vector>
#include <cfloat>
#include <omp.h>
#include "defines.h"
#include "costFunctions.h"
#include "ParallelTempering_detail.h"

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
  template <class T>
  StringDoubleMap fit(std::vector<T> &xdata_in,
                      std::vector<double> &ydata_in,
                      ModelFunction<T> model,
                      std::vector<StringDoubleMap> &initialGuesses,
                      Parameters params = Parameters(),
                      CostFunction costFunction = squaredError,
                      StringDoubleMap extraParameters = StringDoubleMap{}){
    
    //Initialize all the parameters
    int nTemperatures = params.temperatures.size();
    double minError   = DBL_MAX;
    std::vector<StringDoubleMap> allFittingParameters = initialGuesses;
    std::vector<StringDoubleMap> optimalFittingParameters= initialGuesses;
    std::vector<double> optimalErrors(nTemperatures, minError);
    StringDoubleMap bestParameters;
    
    auto start           = std::chrono::high_resolution_clock::now();
    int stepsSameError   = 0;
    bool minErrorChanged = false;
    
    for (int step = 0; step < params.maxIterations; ++step) {
      // Update errors for all temperatures in parallel
      std::vector<double> errors(nTemperatures);
      
#pragma omp parallel for num_threads(params.numThreads)
      for (int tempIdx = 0; tempIdx < nTemperatures; ++tempIdx) {
        errors[tempIdx] = forwardTimeMC(xdata_in, ydata_in, model, allFittingParameters[tempIdx],
                                        params.temperatures[tempIdx], params.jumpSize[tempIdx],
                                        costFunction, extraParameters);
        
        // Update optimal parameters for this temperature
        updateOptimalParameters(tempIdx, errors[tempIdx], optimalErrors, 
                                allFittingParameters, optimalFittingParameters);
      }
      
      if (step % params.printSteps == 0 and step > 0){
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start;
        printStepProgress(step, params.maxIterations, start,
                          elapsed.count(), optimalErrors,
                          optimalFittingParameters);
        start = now;
      }
      
      // Perform swaps
      if (step % params.numStepsSwap == 0 and step > 0) {
        
        tryToSwapTemperatures(params.temperatures, allFittingParameters,
                              optimalFittingParameters, optimalErrors);
        
        // Update all fitting parameters after swaps
        allFittingParameters = optimalFittingParameters;
        
        checkAndUpdateBestParameters(optimalErrors, optimalFittingParameters,
                                     minError, bestParameters, stepsSameError,
                                     minErrorChanged, params.numStepsSwap);
        
        if (stepsSameError >= params.numStepsFinish || minError < params.tolerance) {
          break;
        }
      }
    }
    return bestParameters;
  }
} // namespace ParallelTempering
} // namespace FittingAlgorithms

