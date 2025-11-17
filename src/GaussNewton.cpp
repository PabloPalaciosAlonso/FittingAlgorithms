#include "GaussNewton.h"

namespace FittingAlgorithms{
  namespace GaussNewton{
    
    using vector = Eigen::VectorXd;
    using matrix = Eigen::MatrixXd;
    
    static double EPSILON = sqrt(std::numeric_limits<double>::epsilon());
    

    // Function to convert StringDoubleMap to Eigen::VectorXd
    inline vector mapToEigen(const StringDoubleMap& initialGuesses) {
      
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

  
    // Function to update a StringDoubleMap with values from an Eigen::VectorXd
    inline void updateMapFromEigen(StringDoubleMap& m, const vector& eigenVec) {
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
  
    inline void printParameters(const StringDoubleMap& parameters, int iter) {
      std::cout<<"Iteration "<<iter<<std::endl;
      for (const auto& param : parameters) {
        std::cout << "  " << param.first << " = " << param.second << std::endl;
      }
      std::cout<<"\n";
    }
    
    vector computeResiduals(std::vector<double> &xdata_in,
                            std::vector<double> &ydata_in,
                            ModelFunction model,
                            CostFunction costFunction,
                            StringDoubleMap& fittingParameters,
                            StringDoubleMap& extraParameters){
      
      std::vector<double> y_predV(xdata_in.size()); // Reserva espacio para el resultado

      std::transform(xdata_in.begin(), xdata_in.end(), y_predV.begin(),
                     [&](const double x_element) {
                       return model(x_element, fittingParameters, extraParameters);
                     });
      
      vector residuals(y_predV.size());
      for(int i = 0; i<ydata_in.size(); i++){
        residuals(i) = sqrt(costFunction(ydata_in[i], y_predV[i]));
      }
      return residuals;
    }
    
    // Function to compute the Jacobian matrix numerically using finite differences
    Eigen::MatrixXd computeJacobian(std::vector<double> &xdata_in,
                                    std::vector<double> &ydata_in,
                                    ModelFunction model,
                                    CostFunction costFunction,
                                    StringDoubleMap &paramsMap,
                                    StringDoubleMap &extraParameters) {
      // Create a map to store perturbed parameters
      StringDoubleMap perturbedParams = paramsMap;
      
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
        J.col(col) = (y_predPerturbed - y_predBase) / EPSILON;
        ++col;
      }
      return J;
    }
  
    matrix computePseudoJacobian(std::vector<double> &xdata_in,
                                 std::vector<double> &ydata_in,
                                 ModelFunction model,
                                 CostFunction costFunction,
                                 double regularization,
                                 StringDoubleMap &paramsMap,
                                 StringDoubleMap &extraParameters) {
    
    
      int nParams           = paramsMap.size();
      matrix R              = Eigen::MatrixXd::Identity(nParams, nParams) * regularization;
      matrix J              = computeJacobian(xdata_in, ydata_in, model, costFunction,
                                              paramsMap, extraParameters);
      matrix JTJ            = J.transpose() * J + R;
      Eigen::LDLT<matrix> ldlt(JTJ);
    
      matrix pseudoJ = ldlt.solve(J.transpose());
      return pseudoJ;
    }

    StringDoubleMap computeStandardErrors(std::vector<double> &xdata_in,
                                          std::vector<double> &ydata_in,
                                          ModelFunction model,
                                          CostFunction costFunction,
                                          double regularization,
                                          StringDoubleMap &paramsMap,
                                          StringDoubleMap &extraParameters,
                                          vector residual){
      
      StringDoubleMap errors;
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

    FitResult fit(std::vector<double>& xdata_in,
                  std::vector<double>& ydata_in,
                  ModelFunction model,
                  StringDoubleMap& initialGuesses,
                  Parameters gnParams,
                  CostFunction costFunction,
                  StringDoubleMap extraParameters){
      
      vector fittingParams = mapToEigen(initialGuesses);
      int n                = initialGuesses.size();
      vector y_data        = Eigen::Map<const vector>(ydata_in.data(), ydata_in.size());
      StringDoubleMap fittingParamsMap = initialGuesses;
    
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
  }
}
