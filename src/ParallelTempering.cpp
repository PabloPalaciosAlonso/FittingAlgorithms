#include"ParallelTempering.h"

namespace FittingAlgorithms{
  namespace ParallelTempering {

    inline std::mt19937& getRng() {
      static std::random_device rd;
      static std::mt19937 rng(rd());
      return rng;
    }
    
    inline int generateRandomInt(int first, int last) {
      std::uniform_int_distribution<int> dist(first, last);
      return dist(getRng());
    }
  
    inline double generateRandomDouble(double first, double last) {
      std::uniform_real_distribution<double> dist(first, last);
      return dist(getRng());
    }
    
    inline void printParameters(const std::vector<StringDoubleMap>& optimalFittingParameters) {
      std::cout<<optimalFittingParameters[0].size()<<std::endl;
      for (size_t i = 0; i < optimalFittingParameters.size(); ++i) {
        std::cout << "Temperature " << i + 1 << " parameters:" << std::endl;
        for (const auto& param : optimalFittingParameters[i]) {
          std::cout << "  " << param.first << "= " << param.second<<";" << std::endl;
        }
      }
    }
    
    inline std::string getRandomParameter(const StringDoubleMap& parameters) {
      int nParameters = parameters.size();
      if (nParameters == 0) {
        throw std::runtime_error("The parameters map is empty");
      }

      int randomIndex = generateRandomInt(0, nParameters-1);
    
      auto it = parameters.begin();
      std::advance(it, randomIndex);
    
      return it->first;
    }

    inline StringDoubleMap proposeNewFittingParameters(const StringDoubleMap currentFittingParameters,
                          const double jumpSize){

      
      std::string param                    = getRandomParameter(currentFittingParameters);
      StringDoubleMap newFittingParameters = currentFittingParameters;
      double paramValue                    = currentFittingParameters.at(param);
      double increment                     = generateRandomDouble(-jumpSize, jumpSize);
      
      paramValue                 += increment*paramValue;
      newFittingParameters[param] = paramValue;
      return newFittingParameters;
    }

    inline void updateFittingParameters(StringDoubleMap &fittingParameters,
                   StringDoubleMap newFittingParameters,
                   double oldError,
                   double &newError,
                   double temperature,
                   double &jumpSize){
      
      if (newError < oldError){
        fittingParameters = newFittingParameters;
        jumpSize*=1.01;
      } else {
        double randNum = generateRandomDouble(0.0, 1.0);
        if (randNum<exp(-(newError-oldError)/temperature)){
          fittingParameters = newFittingParameters;
        } else {
          newError = oldError;
          jumpSize*=0.99;
        }
      }
    }
    
    inline void tryToSwapTemperatures(std::vector<double>& temperatures,
                   std::vector<std::map<std::string, double>>& allFittingParameters,
                   std::vector<std::map<std::string, double>>& optimalFittingParameters,
                   std::vector<double>& errors) {
      int numTemperatures = temperatures.size();
      bool shouldTryToSwap = true;
      
      // Select a random starting index
      int startIndex = generateRandomInt(0, numTemperatures - 1);
      
      // Loop through temperature indices with offset from the random start
      for (int i = 0; i < numTemperatures - 1; ++i) {
        // Skip iteration if swapping is not allowed
        if (!shouldTryToSwap) {
          shouldTryToSwap = true;
          continue;
        }
        
        int currentIndex = (i + startIndex) % numTemperatures;
        int nextIndex = (i + startIndex + 1) % numTemperatures;
        
        // Ensure indices are valid
        if (nextIndex < currentIndex) {
          continue;
        }
        
        // Compute the delta to decide the swap
        double delta = (1.0 / temperatures[currentIndex] - 1.0 / temperatures[nextIndex]) *
          (errors[nextIndex] - errors[currentIndex]);
        
        // Swap if conditions are met
        if (delta < 0 || generateRandomDouble(0, 1) < std::exp(-delta)) {
          // Swap optimalFittingParameters
          std::swap(optimalFittingParameters[currentIndex], optimalFittingParameters[nextIndex]);
          
          // Swap errors
          std::swap(errors[currentIndex], errors[nextIndex]);
          
          shouldTryToSwap = false;
        }

        // Update allFittingParameters with new optimal parameters
        allFittingParameters[currentIndex] = optimalFittingParameters[currentIndex];
        allFittingParameters[nextIndex]    = optimalFittingParameters[nextIndex];
      }
    }

    // Helper function to update optimal parameters for a given temperature
    inline void updateOptimalParameters(int tempIdx, double currentError,
                   std::vector<double>& optimalErrors,
                   const std::vector<StringDoubleMap>& allFittingParameters,
                   std::vector<StringDoubleMap>& optimalFittingParameters) {
      if (currentError < optimalErrors[tempIdx]) {
        optimalErrors[tempIdx] = currentError;
        optimalFittingParameters[tempIdx] = allFittingParameters.at(tempIdx);
      }
    }
    
    // Helper function to print step progress
    inline void printStepProgress(int step, int maxSteps,
                 const std::chrono::high_resolution_clock::time_point& start,
                 double elapsedTime,
                 const std::vector<double>& optimalErrors,
                 const std::vector<StringDoubleMap>& optimalFittingParameters) {
      std::cout << "Step " << step << "/" << maxSteps << std::endl;
      std::cout << "Elapsed time: " << elapsedTime << " seconds" << std::endl;
      
      if (!optimalErrors.empty()) {
        std::cout << "Optimal Errors: ";
        for (const auto& error : optimalErrors) {
          std::cout << error << " ";
        }
        std::cout << std::endl;
      }
      
      if (!optimalFittingParameters.empty()) {
        std::cout << "Optimal Parameters:" << std::endl;
        for (size_t i = 0; i < optimalFittingParameters.size(); ++i) {
          std::cout << "Temperature " << i + 1 << " parameters:" << std::endl;
          for (const auto& param : optimalFittingParameters[i]) {
            std::cout << "  " << param.first << "= " << param.second << ";" << std::endl;
          }
        }
      }
    }

    // Helper function to check and update the best parameters
    inline void checkAndUpdateBestParameters(const std::vector<double>& optimalErrors,
                                             const std::vector<StringDoubleMap>& optimalFittingParameters,
                                             double& minError, StringDoubleMap& bestParameters,
                                             int& stepsSameError, bool& minErrorChanged,
                                             const Parameters& params) {
      minErrorChanged = false; // Reset the flag
      for (size_t i = 0; i < optimalErrors.size(); ++i) {
        if (optimalErrors[i] < minError) {
          minError = optimalErrors[i];
          bestParameters = optimalFittingParameters[i];
          stepsSameError = 0;
          minErrorChanged = true;
        }
      }
      
      if (!minErrorChanged) {
        stepsSameError += params.numStepsSwap;
      }
    }
    
    double computeAverageError(const std::vector<double>& xdata_in,
                               const std::vector<double>& ydata_in,
                               const ModelFunction& model,
                               const StringDoubleMap& fittingParameters,
                               const CostFunction& costFunction,                               
                               const StringDoubleMap& extraParameters) {
      
      auto calculateError = [&](const double& x_element, const double& y_actual) {
        double y_pred = model(x_element, fittingParameters, extraParameters);
        return costFunction(y_actual, y_pred);
      };
      
      double totalError = std::transform_reduce(xdata_in.begin(),
                                                xdata_in.end(),
                                                ydata_in.begin(),
                                                0.0,
                                                std::plus<>(),
                                                calculateError);
      
      return totalError / xdata_in.size();
    }
    
    double forwardTimeMC(std::vector<double> &xdata,
                         std::vector<double> &ydata,
                         ModelFunction model,
                         StringDoubleMap &fittingParameters,
                         double temperature,
                         double &jumpSize,
                         CostFunction costFunc,
                         StringDoubleMap &extraParameters){
      
      double oldError           = computeAverageError(xdata, ydata, model,
                                                      fittingParameters,
                                                      costFunc, extraParameters);
      
      StringDoubleMap newFittingParameters = proposeNewFittingParameters(fittingParameters, jumpSize);
      
      double newError             = computeAverageError(xdata, ydata, model,
                                                        newFittingParameters,
                                                        costFunc, extraParameters);

      updateFittingParameters(fittingParameters, newFittingParameters,
                              oldError, newError, temperature, jumpSize);
      
      return newError;
    }
    
    StringDoubleMap fit(std::vector<double> &xdata_in,
                        std::vector<double> &ydata_in,
                        ModelFunction model,
                        std::vector<StringDoubleMap> &initialGuesses,
                        Parameters params,
                        CostFunction costFunction,
                        StringDoubleMap extraParameters){
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
                                       minErrorChanged, params);
          
          if (stepsSameError >= params.numStepsFinish || minError < params.tolerance) {
            break;
          }
        }
      }
      return bestParameters;
    }
    
    StringDoubleMap fit(std::vector<double> &xdata_in,
                        std::vector<double> &ydata_in,
                        ModelFunction model,
                        StringDoubleMap &initialGuess,
                        Parameters params,
                        CostFunction costFunction,
                        StringDoubleMap extraParameters){
      
      std::vector<StringDoubleMap> initialGuesses(params.temperatures.size());
      std::fill(initialGuesses.begin(), initialGuesses.end(), initialGuess);
      return fit(xdata_in, ydata_in, model, initialGuesses, params,
                 costFunction, extraParameters);
    }
  }
}

