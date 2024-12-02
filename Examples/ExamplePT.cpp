#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include "../src/ParallelTempering.h"
#include "../src/utils/costFunctions.h"
#include "../src/utils/defines.h"


using namespace FittingAlgorithms;

// Modelo polinómico
double model(double x, const std::map<std::string, double>& params, const std::map<std::string, double>& extraParams) {
    double a = params.at("a");
    double b = params.at("b");
    double c = params.at("c");
    return a * x * x + b * x + c;
}

int main(int argc, char* argv[]) {
  // Parámetros verdaderos para generar datos
  std::map<std::string, double> trueParameters = {{"a", 2.5}, {"b", 1.2}, {"c", 3.33}};

  // Generar los datos
  std::vector<double> xdata(100);
  std::vector<double> ydata(100);
  for (size_t i = 0; i < xdata.size(); ++i) {
    xdata[i] = static_cast<double>(i) / 10.0; // x en el rango [0, 10]
    ydata[i] = model(xdata[i], trueParameters, {});
  }
 
  ParallelTempering::PTParameters ptParams;
  ptParams.maxIterations  = 10000;
  ptParams.temperatures   = {1e-6, 1e-4, 1e-2, 1e0, 1e2};
  ptParams.jumpSize       = {0.001, 0.001, 0.001, 0.001, 0.001};
  ptParams.numStepsSwap   = 1000;
  ptParams.numStepsFinish = 50000;
  ptParams.maxIterations  = 100000;
  ptParams.tolerance      = 1e-8;
  ptParams.printSteps     = 5000;
  
  // Suposiciones iniciales
  StringDoubleMap initialGuesses = {{"a", 1.0}, {"b", 1.0}, {"c", 6.0}};

  auto result = ParallelTempering::fit<double>(xdata, ydata, model,
                                               initialGuesses, ptParams);
  
  // Imprimir resultados
    std::cout << "Fitted Parameters:\n";
    for (const auto& param : result) {
      std::cout << param.first << ": " << param.second << "\n";
    }
    
    std::cout << "Target Parameters:\n";
    for (const auto& param : trueParameters) {
      std::cout << param.first << ": " << param.second << "\n";
    }
    
    return 0;
}
