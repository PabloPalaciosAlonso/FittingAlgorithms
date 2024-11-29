#pragma once
#include <map>
#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <functional>
#include "utils/costFunctions.h"

namespace FittingAlgorithms {
  namespace GaussNewton {
    // Declaraciones de las funciones
    
    using vector = Eigen::VectorXd;
    using matrix = Eigen::MatrixXd;
    
    using CostFunction = std::function<double(const double, const double)>;
    
    template <typename T>
    using ModelFunction = std::function<double(const T,
                                               const std::map<std::string, double>&,
                                               const std::map<std::string, double>&)>;
  
    constexpr double EPSILON = sqrt(std::numeric_limits<double>::epsilon());
    
    struct GNParameters{
      int maxIterations     = 100;
      double tolerance      = 1e-5;
      double regularization = 1e-6;
      int printSteps        = 10;
    };
    
    struct FitResult {
      std::map<std::string, double> parameters;
      std::map<std::string, double> errors;
    };
        
    Eigen::VectorXd mapToEigen(const std::map<std::string, double>& params);
    void updateMapFromEigen(std::map<std::string, double>& params, const Eigen::VectorXd& values);
    void printParameters(const std::map<std::string, double>& params, int precision = 6);

    template <typename T>
    Eigen::MatrixXd computeJacobian(std::vector<T> &xdata_in,
                                    std::vector<double> &ydata_in,
                                    ModelFunction<T> model,
                                    CostFunction costFunction,
                                    std::map<std::string, double> &paramsMap,
                                    std::map<std::string, double> &extraParameters);
    
    template <typename T>
    Eigen::MatrixXd computePseudoJacobian(std::vector<T>& xdata_in,
                                          std::vector<double>& ydata_in,
                                          ModelFunction<T> model,
                                          CostFunction costFunction,
                                          double regularization,
                                          std::map<std::string, double>& paramsMap,
                                          std::map<std::string, double>& extraParameters);

    template<typename T>
    std::map<std::string, double> computeStandardErrors(std::vector<T> &xdata_in,
                                                        std::vector<double> &ydata_in,
                                                        ModelFunction<T> model,
                                                        CostFunction costFunction,
                                                        double regularization,
                                                        std::map<std::string, double> &paramsMap,
                                                        std::map<std::string, double> &extraParameters,
                                                        vector residual);

    template <typename T>
    FitResult fit(std::vector<T>& xdata_in,
                  std::vector<double>& ydata_in,
                  ModelFunction<T> model,
                  std::map<std::string, double>& initialGuesses,
                  GNParameters gnParams = GNParameters(),
                  CostFunction costFunction = squaredError,
                  std::map<std::string, double> extraParameters = std::map<std::string, double>{});

    FitResult fitScalar(std::vector<double>& xdata_in,
                        std::vector<double>& ydata_in,
                        ModelFunction<double> model,
                        std::map<std::string, double>& initialGuesses,
                        GNParameters gnParams = GNParameters(),
                        CostFunction costFunction = squaredError,
                        std::map<std::string, double> extraParameters = std::map<std::string, double>{});

    FitResult fitVector(std::vector<std::vector<double>>& xdata_in,
                        std::vector<double>& ydata_in,
                        ModelFunction<std::vector<double>> model,
                        std::map<std::string, double>& initialGuesses,
                        GNParameters gnParams = GNParameters(),
                        CostFunction costFunction = squaredError,
                        std::map<std::string, double> extraParameters = std::map<std::string, double>{});
    
  } 
} 
