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
    template <typename T1, typename Model>
      Eigen::MatrixXd computeJacobian(
                                      Model model,
                                      std::vector<T1>& xdata_in,
                                      std::vector<double>& ydata_in,
                                      std::map<std::string, double>& paramsMap,
                                      std::map<std::string, double>& extraParameters);
    
    template <typename T1, typename Model>
      Eigen::MatrixXd computePseudoJacobian(
                                            Model model,
                                            std::vector<T1>& xdata_in,
                                            std::vector<double>& ydata_in,
                                            GNParameters params,
                                            std::map<std::string, double>& paramsMap,
                                            std::map<std::string, double>& extraParameters);
    
    template <typename T1, typename Model>
      std::map<std::string, double> computeStandardErrors(
                                                          Model model,
                                                          std::vector<T1>& xdata_in,
                                                          std::vector<double>& ydata_in,
                                                          GNParameters params,
                                                          std::map<std::string, double>& paramsMap,
                                                          std::map<std::string, double>& extraParameters,
                                                          Eigen::VectorXd residual);

    template <typename T1>
    FitResult fit(std::vector<T1>& xdata_in,
                  std::vector<double>& ydata_in,
                  GNParameters gnParams,
                  std::function<std::vector<double>(const std::vector<T1>&, const std::vector<double>&,
                                                    const std::map<std::string, double>&, const std::map<std::string, double>&)> model,
                  std::map<std::string, double>& initialGuesses,
                  std::map<std::string, double>& extraParameters);

    
    FitResult fitScalar(std::vector<double>& xdata_in,
                        std::vector<double>& ydata_in,
                        GNParameters gnParams,
                        std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&,
                                                          const std::map<std::string, double>&, const std::map<std::string, double>&)> model,
                        std::map<std::string, double>& initialGuesses,
                        std::map<std::string, double>& extraParameters);

    FitResult fitVector(std::vector<std::vector<double>>& xdata_in,
                        std::vector<double>& ydata_in,
                        GNParameters gnParams,
                        std::function<std::vector<double>(const std::vector<std::vector<double>>&, const std::vector<double>&,
                                                          const std::map<std::string, double>&, const std::map<std::string, double>&)> model,
                        std::map<std::string, double>& initialGuesses,
                        std::map<std::string, double>& extraParameters);
    
    
  } 
} 
