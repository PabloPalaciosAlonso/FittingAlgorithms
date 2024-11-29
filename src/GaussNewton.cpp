#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include"GaussNewton.h"
#include"utils/costFunctions.h"

namespace FittingAlgorithms{
  namespace GaussNewton{

    // Function to convert std::map<std::string, double> to Eigen::VectorXd
    vector mapToEigen(const std::map<std::string, double>& initialGuesses) {
      
      std::size_t total_size = initialGuesses.size();
    
      // Create an Eigen::VectorXd with the size of the map
      vector eigenVec(total_size);
      
      // Populate the Eigen::VectorXd with values from the map
      std::size_t index = 0;
      for (const auto& pair : initialGuesses) {
        eigenVec(index++) = pair.second; // Store the map value in the vector
      }
    
      return eigenVec;
    }

  
    // Function to update a std::map<std::string, double> with values from an Eigen::VectorXd
    void updateMapFromEigen(std::map<std::string, double>& m, const vector& eigenVec) {
      // Ensure that the size of the Eigen::VectorXd matches the size of the std::map
      if (m.size() != int(eigenVec.size())) {
        std::cerr << "Error: The size of the vector (" << eigenVec.size()
                  << ") does not match the size of the map (" << m.size() << ")" << std::endl;
        return;
      }
    
      // Iterator to traverse the map
      auto it = m.begin();
    
      // Iterate through the Eigen::VectorXd values and update the map
      for (int i = 0; i < eigenVec.size(); ++i) {
        it->second = eigenVec(i);  // Update the value in the map
        ++it;  // Move to the next element in the map
      }
    }
  
    void printParameters(const std::map<std::string, double>& parameters, int iter) {
      std::cout<<"Iteration "<<iter<<std::endl;
      for (const auto& param : parameters) {
        std::cout << "  " << param.first << " = " << param.second << std::endl;
      }
      std::cout<<"\n";
    }

    // template<typename T1, typename Model, typename Jacobian>
    // std::map<std::string, double> fitParams(std::vector<T1> &xdata_in,
    //                                         std::vector<double> &ydata_in,
    //                                         GNParameters params,
    //                                         Model model,
    //                                         Jacobian jacobian,
    //                                         std::map<std::string, double> &initialGuesses,
    //                                         std::map<std::string, double> &extraParameters) {
    
    //   vector fittingParams  = mapToEigen(initialGuesses);
    //   int n = initialGuesses.size();
    //   matrix regularization = Eigen::MatrixXd::Identity(n, n) * params.regularization; 
    //   vector y_data         = Eigen::Map<const vector>(ydata_in.data(), ydata_in.size());
    //   std::map<std::string, double> fittingParamsMap = initialGuesses;
    
    //   for (int i = 0; i < params.maxIterations; ++i) {
    //     std::vector<double> y_predV = model(xdata_in, fittingParamsMap, extraParameters);
    //     vector y_pred               = Eigen::Map<const vector>(y_predV.data(),
    //                                                            y_predV.size());
      
    //     vector residual = y_data - y_pred;
    //     matrix J       = jacobian(xdata_in, fittingParamsMap, extraParameters);      
    //     matrix JTJ     = J.transpose() * J + regularization;
    //     matrix JTJ_inv = JTJ.inverse();
    //     matrix pseudoJ = JTJ_inv * J.transpose();
    //     vector delta_params = pseudoJ * residual;

    //     fittingParams += delta_params;
    //     updateMapFromEigen(fittingParamsMap, fittingParams);

    //     if (i>0 and i%params.printSteps == 0)
    //       printParameters(fittingParamsMap, i);
      
    //     if (delta_params.norm() < params.tolerance) {
    //       break;
    //     }
    //   }    
    //   return fittingParamsMap;
    // }


    
    template <typename T>
    vector computeResiduals(std::vector<T> &xdata_in,
                            std::vector<double> &ydata_in,
                            ModelFunction<T> model,
                            CostFunction costFunction,
                            std::map<std::string, double>& fittingParameters,
                            std::map<std::string, double>& extraParameters){
      
      std::vector<double> y_predV(xdata_in.size()); // Reserva espacio para el resultado

      std::transform(xdata_in.begin(), xdata_in.end(), y_predV.begin(),
                     [&](const T x_element) {
                       return model(x_element, fittingParameters, extraParameters);
                     });
      
      vector residuals(y_predV.size());
      for(int i = 0; i<ydata_in.size(); i++){
        residuals(i) = sqrt(costFunction(ydata_in[i], y_predV[i]));
      }
      return residuals;
    }
    
    
    // Function to compute the Jacobian matrix numerically using finite differences
    template <typename T>
    Eigen::MatrixXd computeJacobian(std::vector<T> &xdata_in,
                                    std::vector<double> &ydata_in,
                                    ModelFunction<T> model,
                                    CostFunction costFunction,
                                    std::map<std::string, double> &paramsMap,
                                    std::map<std::string, double> &extraParameters) {
      // Create a map to store perturbed parameters
      std::map<std::string, double> perturbedParams = paramsMap;
      
      vector y_predBase = computeResiduals(xdata_in, ydata_in,
                                           model, costFunction,
                                           paramsMap, extraParameters);
      int m = y_predBase.size();
      int n = paramsMap.size();
      Eigen::MatrixXd J(m, n);
    
      int col = 0;
      for (const auto& param : paramsMap) {
        double originalValue = param.second;
        perturbedParams[param.first] = originalValue + EPSILON;
        vector y_predPerturbed = computeResiduals(xdata_in, ydata_in,
                                                  model, costFunction,
                                                  perturbedParams, extraParameters);
      
        perturbedParams[param.first] = originalValue;
        for (int i = 0; i < y_predBase.size(); ++i) {
          J(i, col) = (y_predPerturbed[i] - y_predBase[i]) / EPSILON;
        }
        ++col;
      }
      return J;
    }
  
    template<typename T>
    matrix computePseudoJacobian(std::vector<T> &xdata_in,
                                 std::vector<double> &ydata_in,
                                 ModelFunction<T> model,
                                 CostFunction costFunction,
                                 double regularization,
                                 std::map<std::string, double> &paramsMap,
                                 std::map<std::string, double> &extraParameters) {
    
    
      int nParams           = paramsMap.size();
      matrix R              = Eigen::MatrixXd::Identity(nParams, nParams) * regularization;
      matrix J              = computeJacobian(xdata_in, ydata_in, model, costFunction,
                                              paramsMap, extraParameters);
      matrix JTJ            = J.transpose() * J + R;
      Eigen::LDLT<matrix> ldlt(JTJ);
    
      matrix pseudoJ = ldlt.solve(J.transpose());
      return pseudoJ;
    }

    template<typename T>
    std::map<std::string, double> computeStandardErrors(std::vector<T> &xdata_in,
                                                        std::vector<double> &ydata_in,
                                                        ModelFunction<T> model,
                                                        CostFunction costFunction,
                                                        double regularization,
                                                        std::map<std::string, double> &paramsMap,
                                                        std::map<std::string, double> &extraParameters,
                                                        vector residual){
    
      std::map<std::string, double> errors;
      int nParams        = paramsMap.size();
      double residualSum = residual.squaredNorm();
      double variance    = residualSum / (ydata_in.size() - nParams);
      matrix R           = Eigen::MatrixXd::Identity(nParams, nParams) * regularization;
      matrix J           = computeJacobian(xdata_in, ydata_in, model,
                                           costFunction, paramsMap,
                                           extraParameters);
      
      matrix JTJ         = J.transpose() * J + R;
      matrix JTJ_inv     = JTJ.inverse();
    
      for (int i = 0; i < nParams; ++i) {
        errors[std::next(paramsMap.begin(), i)->first] = std::sqrt(variance * JTJ_inv(i, i));
      }
      return errors;
    }

    template <typename T>
    FitResult fit(std::vector<T>& xdata_in,
                  std::vector<double>& ydata_in,
                  GNParameters gnParams,
                  ModelFunction<T> model,
                  CostFunction costFunction,
                  std::map<std::string, double>& initialGuesses,
                  std::map<std::string, double>& extraParameters){
      
      vector fittingParams = mapToEigen(initialGuesses);
      int n                = initialGuesses.size();
    
      vector y_data        = Eigen::Map<const vector>(ydata_in.data(), ydata_in.size());
      std::map<std::string, double> fittingParamsMap = initialGuesses;
    
      vector residual;
      for (int i = 0; i < gnParams.maxIterations; ++i) {
        
        residual = computeResiduals(xdata_in, ydata_in,
                                    model, costFunction,
                                    fittingParamsMap, extraParameters);
        
        matrix pseudoJ      = computePseudoJacobian(xdata_in, ydata_in,
                                                    model, costFunction,
                                                    gnParams.regularization,
                                                    fittingParamsMap,
                                                    extraParameters);
        vector delta_params = - pseudoJ * residual;
        double currentError = (delta_params.array() / fittingParams.array()).matrix().norm()/n;
      
        fittingParams += delta_params;
        updateMapFromEigen(fittingParamsMap, fittingParams);
      
        if (i>0 and i%gnParams.printSteps == 0)
          printParameters(fittingParamsMap, i);
      
        if (currentError < gnParams.tolerance) {
          break;
        }
      }

      auto errors = computeStandardErrors(xdata_in, ydata_in,
                                          model, costFunction,
                                          gnParams.regularization,
                                          fittingParamsMap,
                                          extraParameters,
                                          residual);
    
      return FitResult{fittingParamsMap, errors};
    }

    template <typename T>
    FitResult fit(std::vector<T>& xdata_in,
                  std::vector<double>& ydata_in,
                  GNParameters gnParams,
                  ModelFunction<T> model,
                  std::map<std::string, double>& initialGuesses,
                  std::map<std::string, double>& extraParameters){
      return fit(xdata_in, ydata_in, gnParams, model,
                 squaredError, initialGuesses, extraParameters);
    }
    
    FitResult fitScalar(std::vector<double>& xdata_in,
                        std::vector<double>& ydata_in,
                        GNParameters gnParams,
                        ModelFunction<double> model,
                        CostFunction costFunction,
                        std::map<std::string, double>& initialGuesses,
                        std::map<std::string, double>& extraParameters){
      return fit<double>(xdata_in, ydata_in, gnParams, model,
                         costFunction, initialGuesses, extraParameters);
    }
    
    FitResult fitScalar(std::vector<double>& xdata_in,
                        std::vector<double>& ydata_in,
                        GNParameters gnParams,
                        ModelFunction<double> model,
                        std::map<std::string, double>& initialGuesses,
                        std::map<std::string, double>& extraParameters){
      return fit<double>(xdata_in, ydata_in, gnParams, model,
                         squaredError, initialGuesses, extraParameters);
}

    FitResult fitVector(std::vector<std::vector<double>>& xdata_in,
                        std::vector<double>& ydata_in,
                        GNParameters gnParams,
                        ModelFunction<std::vector<double>> model,
                        CostFunction costFunction,
                        std::map<std::string, double>& initialGuesses,
                        std::map<std::string, double>& extraParameters){
      return fit<std::vector<double>>(xdata_in, ydata_in, gnParams, model,
                                      costFunction, initialGuesses, extraParameters);
    }

    FitResult fitVector(std::vector<std::vector<double>>& xdata_in,
                        std::vector<double>& ydata_in,
                        GNParameters gnParams,
                        ModelFunction<std::vector<double>> model,
                        std::map<std::string, double>& initialGuesses,
                        std::map<std::string, double>& extraParameters){
      return fit<std::vector<double>>(xdata_in, ydata_in, gnParams, model,
                                      squaredError, initialGuesses, extraParameters);
    }
  }
}
