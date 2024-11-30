#pragma once
#include <map>
#include <string>
#include <Eigen/Dense>
#include <iostream>
#include "utils/defines.h"
#include "utils/costFunctions.h"

namespace FittingAlgorithms {
  namespace GaussNewton {
        
    using vector = Eigen::VectorXd;
    using matrix = Eigen::MatrixXd;
      
    constexpr double EPSILON = sqrt(std::numeric_limits<double>::epsilon());
    
    struct GNParameters{
      int maxIterations     = 100;
      int printSteps        = 10;
      double tolerance      = 1e-5;
      double regularization = 1e-6;
      
    };
    
    struct FitResult {
      StringDoubleMap parameters;
      StringDoubleMap errors;
    };
        
    Eigen::VectorXd mapToEigen(const StringDoubleMap& params);
    void updateMapFromEigen(StringDoubleMap& params, const Eigen::VectorXd& values);
    void printParameters(const StringDoubleMap& params, int precision = 6);

    template <typename T>
    vector computeResiduals(std::vector<T> &xdata_in,
                            std::vector<double> &ydata_in,
                            ModelFunction<T> model,
                            CostFunction costFunction,
                            StringDoubleMap& fittingParameters,
                            StringDoubleMap& extraParameters);
    
    template <typename T>
    Eigen::MatrixXd computeJacobian(std::vector<T> &xdata_in,
                                    std::vector<double> &ydata_in,
                                    ModelFunction<T> model,
                                    CostFunction costFunction,
                                    StringDoubleMap &paramsMap,
                                    StringDoubleMap &extraParameters);
    
    template <typename T>
    Eigen::MatrixXd computePseudoJacobian(std::vector<T>& xdata_in,
                                          std::vector<double>& ydata_in,
                                          ModelFunction<T> model,
                                          CostFunction costFunction,
                                          double regularization,
                                          StringDoubleMap& paramsMap,
                                          StringDoubleMap& extraParameters);

    template<typename T>
    StringDoubleMap computeStandardErrors(std::vector<T> &xdata_in,
                                                        std::vector<double> &ydata_in,
                                                        ModelFunction<T> model,
                                                        CostFunction costFunction,
                                                        double regularization,
                                                        StringDoubleMap &paramsMap,
                                                        StringDoubleMap &extraParameters,
                                                        vector residual);

    template <typename T>
    FitResult fit(std::vector<T>& xdata_in,
                  std::vector<double>& ydata_in,
                  ModelFunction<T> model,
                  StringDoubleMap& initialGuesses,
                  GNParameters gnParams,
                  CostFunction costFunction,
                  StringDoubleMap extraParameters);
  } 
} 
