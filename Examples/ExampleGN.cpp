#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include "../src/GaussNewton.h"
#include "../src/utils/costFunctions.h"
#include "../src/utils/defines.h"

// Modelo polinómico
double model(double x, const std::map<std::string, double>& params,
             const std::map<std::string, double>& extraParams) {
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
  
  // Configurar parámetros de Gauss-Newton
  FittingAlgorithms::GaussNewton::Parameters gnParams;
  gnParams.maxIterations = 10000;
  gnParams.tolerance = 1e-8;
  gnParams.printSteps = 50;
  gnParams.regularization = 1e-5;

  // Suposiciones iniciales
  std::map<std::string, double> initialGuesses = {{"a", 1.0}, {"b", 1.0}, {"c", 6.0}};

  //  FittingAlgorithms::ModelFunction<double> myModel = model;

  auto result = FittingAlgorithms::GaussNewton::fit<double>(xdata, ydata, model,
                                                            initialGuesses, gnParams);
  
  // Imprimir resultados
    std::cout << "Fitted Parameters:\n";
    for (const auto& param : result.parameters) {
      std::cout << param.first << ": " << param.second << "\n";
    }
    
    std::cout << "Target Parameters:\n";
    for (const auto& param : trueParameters) {
      std::cout << param.first << ": " << param.second << "\n";
    }
    
    return 0;
}
