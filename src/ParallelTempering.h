#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <vector>
#include <cfloat>
#include <omp.h>
#include "utils/defines.h"
#include "utils/randomNumbers.h"
#include "utils/costFunctions.h"

namespace FittingAlgorithms{
  namespace ParallelTempering {
    
    struct PTParameters{
      std::vector<double> temperatures;
      std::vector<double> jumpSize;
      int numStepsSwap;
      int numStepsFinish;
      int maxIterations;
      double tolerance = 0;
      int printSteps;
      int numThreads = omp_get_max_threads();
    };
  
    template<typename T2>
    void printParameters(const std::vector<std::map<std::string, T2>>& optimalFittingParameters);
  
    template<typename T>
    std::string getRandomParameter(const std::map<std::string, T>& parameters);
  
  
    template<typename T1, typename T2, typename Cost>
    double forwardTimeMC(std::vector<T1> &xdata,
                         std::vector<T2> &ydata,
                         Cost costFunc,
                         double temperature,
                         double &jumpSize,
                         std::map<std::string, T2> &fittingParameters,
                         std::map<std::string, T2> &extraParameters);
  
    template<typename T>
    void tryToSwapTemperatures(std::vector<double> &temperatures,
                               std::vector<std::map<std::string, T>> &allFittingParameters,
                               std::vector<std::map<std::string, T>> &optimalFittingParameters,
                               std::vector<double> &errors);
  
  
    template<typename T>
    StringDoubleMap fit(std::vector<T> &xdata_in,
                        std::vector<double> &ydata_in,
                        ModelFunction<T> model,
                        std::vector<StringDoubleMap> &initialGuesses,
                        PTParameters ptParams = PTParameters(),
                        CostFunction costFunction = squaredError,
                        StringDoubleMap extraParameters = StringDoubleMap{});
    
    template<typename T>
    StringDoubleMap fit(std::vector<T> &xdata,
                        std::vector<double> &ydata,
                        ModelFunction<T> model,
                        StringDoubleMap &initialGuesses,
                        PTParameters ptParams = PTParameters(),
                        CostFunction costFunction = squaredError,
                        StringDoubleMap &extraParameters = StringDoubleMap{});

  }
}

