#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <vector>
#include <cfloat>
#include <omp.h>
#include"utils/randomNumbers.h"
#include"utils/defines.h"


namespace parallelTempering {

  struct PTParameters{
    std::vector<double> temperatures;
    std::vector<double> jumpSize;
    int numberStepsSwap;
    int maximumNumberSteps;
    int numberStepsFinish;
    double errorBreak = 0;
  };
  
  template<typename T2>
  void printParameters(const std::vector<std::map<std::string, T2>>& optimalFittingParameters) {
    for (size_t i = 0; i < optimalFittingParameters.size(); ++i) {
      std::cout << "Temperature " << i + 1 << " parameters:" << std::endl;
      for (const auto& param : optimalFittingParameters[i]) {
        std::cout << "  " << param.first << "= " << param.second<<";" << std::endl;
      }
    }
  }
  
  template<typename T>
  std::string getRandomParameter(const std::map<std::string, T>& parameters) {
    int nParameters = parameters.size();
    if (nParameters == 0) {
      throw std::runtime_error("The parameters map is empty");
    }

    int randomIndex = generateRandomInt(0, nParameters-1);
    
    auto it = parameters.begin();
    std::advance(it, randomIndex);
    
    return it->first;
  }
  
  
  template<typename T1, typename T2, typename Cost>
  double forwardTimeMC(std::vector<T1> &xdata,
                       std::vector<T2> &ydata,
                       Cost costFunc,
                       double temperature,
                       double &jumpSize,
                       std::map<std::string, T2> &fittingParameters,
                       std::map<std::string, T2> &extraParameters){

    
    double oldError           = costFunc(xdata, ydata, fittingParameters, extraParameters);
    std::string param         = getRandomParameter(fittingParameters);
    auto newFittingParameters = fittingParameters;
    T2 paramValue             = fittingParameters[param];
    double increment          = generateRandomDouble(-jumpSize, jumpSize);

    paramValue                 += increment*paramValue;
    newFittingParameters[param] = paramValue;
    double newError             = costFunc(xdata, ydata, newFittingParameters, extraParameters);
    
    if (newError < oldError){
      fittingParameters = newFittingParameters;
      jumpSize*=1.01;
    } else {
      double randNumber = generateRandomDouble(0.0, 1.0);
      if (randNumber<exp(-(newError-oldError)/temperature)){
        fittingParameters = newFittingParameters;
      } else {
        newError = oldError;
        jumpSize*=0.99;
      }
    }
    return newError;  
  }
  
  template<typename T>
  void tryToSwapTemperatures(std::vector<double> &temperatures,
                             std::vector<std::map<std::string, T>> &allFittingParameters,
                             std::vector<std::map<std::string, T>> &optimalFittingParameters,
                             std::vector<double> &errors){
    int nTemperatures = temperatures.size();
    bool tryToSwap    = true;
    int n0            = generateRandomInt(0, nTemperatures-1);
    fori(0,nTemperatures-1){
      if (!tryToSwap){
        tryToSwap = true;
        continue;
      }
      
      int imod = (i+n0)%nTemperatures;
      int jmod = (i+n0+1)%nTemperatures;
      if (jmod<imod) continue;

      double delta = (1 / temperatures[imod] - 1 / temperatures[jmod]) * (errors[jmod] - errors[imod]);
      if (delta < 0 or generateRandomDouble(0,1) < exp(-delta)){
        auto tmp = optimalFittingParameters[imod];
        optimalFittingParameters[imod] = optimalFittingParameters[jmod];
        optimalFittingParameters[jmod] = tmp;
        
        double tmp2 = errors[imod];
        errors[imod]   = errors[jmod];
        errors[jmod]   = tmp2;
        tryToSwap = false;
      }
      allFittingParameters[imod] = optimalFittingParameters[imod];
      allFittingParameters[jmod] = optimalFittingParameters[jmod];
    }
  }
  
  
  template<typename T1, typename T2, typename Cost>
  std::map<std::string, T2> fitData(std::vector<T1> &xdata,
                                    std::vector<T2> &ydata,
                                    Cost costFunc,
                                    PTParameters params,
                                    std::vector<std::map<std::string, T2>> &initialGuesses,
                                    std::map<std::string, T2> &extraParameters){
    
    int nTemperatures = params.temperatures.size();
    std::vector<std::map<std::string, T2>> allFittingParameters(nTemperatures);
    std::vector<std::map<std::string, T2>> optimalFittingParameters(nTemperatures);
    std::vector<double> optimalErrors(nTemperatures);
    std::map<std::string, T2> bestParameters;
    double minError = DBL_MAX;
        
    fori(0,nTemperatures){
      allFittingParameters[i] = initialGuesses[i];
      optimalErrors[i]        = DBL_MAX;
    }
    
    auto start           = std::chrono::high_resolution_clock::now();
    int stepsSameError   = 0;
    bool minErrorChanged = false;
    
    fori(0, params.maximumNumberSteps){
      std::vector<double> error(nTemperatures);
      #pragma omp parallel for
      forj(0, nTemperatures){
        error[j] = forwardTimeMC(xdata, ydata,
                                 costFunc,
                                 params.temperatures[j],
                                 params.jumpSize[j],
                                 allFittingParameters[j],
                                 extraParameters);
        if (error[j]<optimalErrors[j]){
          optimalErrors[j]            = error[j];
          optimalFittingParameters[j] = allFittingParameters[j];
        }
      }
      
      if (i%params.numberStepsSwap == 0){
        std::cout<<"Step "<<i<<"/"<<params.maximumNumberSteps<<std::endl;
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
        start = now;
        
        tryToSwapTemperatures(params.temperatures,
                              allFittingParameters,
                              optimalFittingParameters,
                              optimalErrors);
        
        allFittingParameters = optimalFittingParameters;
        for (int ll = 0; ll<optimalErrors.size(); ll++){
          std::cout<<optimalErrors[ll]<<" ";
          if (optimalErrors[ll] < minError){
            minError       = optimalErrors[ll];
            bestParameters = optimalFittingParameters[ll];
            stepsSameError = 0;
            minErrorChanged = true;
          }
        }
        if (minErrorChanged == false) {
          stepsSameError+=params.numberStepsSwap;
        }
        minErrorChanged = false;
        std::cout<<std::endl;
        printParameters(optimalFittingParameters);
        
        std::cout<<stepsSameError<<std::endl;
        if (stepsSameError >= params.numberStepsFinish){
          break;
        }
        if (minError< params.errorBreak) break;
      }
    }
    return bestParameters;
  }
  

  template<typename T1, typename T2, typename Cost>
  std::map<std::string, T2> fitData(std::vector<T1> &xdata,
                                    std::vector<T2> &ydata,
                                    Cost costFunc,
                                    PTParameters params,
                                    std::map<std::string, T2> &initialGuess,
                                    std::map<std::string, T2> &extraParameters){

    std::vector<std::map<std::string, T2>> initialGuesses(params.temperatures.size());
    std::fill(initialGuesses.begin(), initialGuesses.end(), initialGuess);
    return fitData(xdata, ydata, costFunc, params,
                   initialGuesses, extraParameters);

  }
}
