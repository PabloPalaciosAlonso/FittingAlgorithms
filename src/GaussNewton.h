#pragma once
#include <map>
#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <functional>

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
      int maxIterations;
      double tolerance;
      double regularization = 1e-8;
      int printSteps = 10;
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
                  GNParameters gnParams,
                  ModelFunction<T> model,
                  CostFunction costFunction,
                  std::map<std::string, double>& initialGuesses,
                  std::map<std::string, double>& extraParameters);

    template <typename T>
    FitResult fit(std::vector<T>& xdata_in,
                  std::vector<double>& ydata_in,
                  GNParameters gnParams,
                  ModelFunction<T> model,
                  std::map<std::string, double>& initialGuesses,
                  std::map<std::string, double>& extraParameters);
    
    FitResult fitScalar(std::vector<double>& xdata_in,
                        std::vector<double>& ydata_in,
                        GNParameters gnParams,
                        ModelFunction<double> model,
                        CostFunction costFunction,
                        std::map<std::string, double>& initialGuesses,
                        std::map<std::string, double>& extraParameters);

    FitResult fitScalar(std::vector<double>& xdata_in,
                        std::vector<double>& ydata_in,
                        GNParameters gnParams,
                        ModelFunction<double> model,
                        std::map<std::string, double>& initialGuesses,
                        std::map<std::string, double>& extraParameters);
    
    FitResult fitVector(std::vector<std::vector<double>>& xdata_in,
                        std::vector<double>& ydata_in,
                        GNParameters gnParams,
                        ModelFunction<std::vector<double>> model,
                        CostFunction costFunction,
                        std::map<std::string, double>& initialGuesses,
                        std::map<std::string, double>& extraParameters);
    
    FitResult fitVector(std::vector<std::vector<double>>& xdata_in,
                        std::vector<double>& ydata_in,
                        GNParameters gnParams,
                        ModelFunction<std::vector<double>> model,
                        std::map<std::string, double>& initialGuesses,
                        std::map<std::string, double>& extraParameters);
    
  } 
} 
