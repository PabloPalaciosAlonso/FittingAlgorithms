#include <vector>
#include <map>
#include <string>
#include "defines.h"
#include "matrixAlgebra.h"

namespace FittingAlgorithms{
  namespace GaussNewton{
    static double EPSILON = sqrt(std::numeric_limits<double>::epsilon());

    inline std::vector<double> vectorToSTD(vector &v) {
      return std::vector<double>(&v[0], &v[0] + v.size());
    }

    template<class T>
    vector computeResiduals(std::vector<T> &xdata_in,
                            std::vector<double> &ydata_in,
                            ModelFunction<T> model,
                            CostFunction costFunction,
                            StringDoubleMap& fittingParameters,
                            StringDoubleMap& extraParameters){
      
      std::vector<double> y_predV(xdata_in.size());
      
      std::transform(xdata_in.begin(), xdata_in.end(), y_predV.begin(),
                     [&](const T x_element) {
                       return model(x_element, fittingParameters, extraParameters);
                     });
      
      vector residuals(y_predV.size());
      for(int i = 0; i<ydata_in.size(); i++){
        residuals(i) =  sqrt(costFunction(ydata_in[i], y_predV[i]));
        if ((ydata_in[i]-y_predV[i])<0){
          residuals(i)*=-1;
        }
      }
      return residuals;
    }

    template<class T>
    matrix computeJacobian(std::vector<T> &xdata_in,
                           std::vector<double> &ydata_in,
                           ModelFunction<T> model,
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
      matrix J(m, n);
      
      int col = 0;
      for (const auto& param : paramsMap) {
        double originalValue = param.second;
        perturbedParams[param.first] = originalValue + EPSILON;
        vector y_predPerturbed = computeResiduals(xdata_in, ydata_in,
                                                  model, costFunction,
                                                  perturbedParams, extraParameters);
        
        perturbedParams[param.first] = originalValue;
        for(int row = 0; row<m; row++){
          J(row, col) = (y_predPerturbed(row) - y_predBase(row)) / EPSILON;
        }
        ++col;
      }
      return J;
    }

    template<class T>
    matrix computePseudoJacobian(std::vector<T> &xdata_in,
                                 std::vector<double> &ydata_in,
                                 ModelFunction<T> model,
                                 CostFunction costFunction,
                                 double regularization,
                                 StringDoubleMap &paramsMap,
                                 StringDoubleMap &extraParameters){
      
      
      int m = ydata_in.size();
      int n = paramsMap.size();
           
      matrix J = computeJacobian(xdata_in, ydata_in, model,
                                 costFunction, paramsMap,
                                 extraParameters);
      
      
      matrix JT  = transpose(J);      
      matrix JTJ = matrixProduct(JT, J);
      
      for(int i = 0; i < n; ++i)
        JTJ(i, i) += regularization;
      
      matrix pseudoJ = solve(JTJ, JT);
      return pseudoJ;
    }

    template<class T>
    std::vector<double> computeParametersIncrement(std::vector<T> &xdata_in,
                                                   std::vector<double> &ydata_in,
                                                   ModelFunction<T> model,
                                                   CostFunction costFunction,
                                                   double regularization,
                                                   StringDoubleMap &fittingParamsMap,
                                                   StringDoubleMap &extraParameters) {
      
      
      vector residual = computeResiduals(xdata_in, ydata_in,
                                         model, costFunction,
                                         fittingParamsMap, extraParameters);
      
      matrix pseudoJ  = computePseudoJacobian(xdata_in, ydata_in,
                                              model, costFunction,
                                              regularization,
                                              fittingParamsMap,
                                              extraParameters);
      
      vector delta_params = matrixProduct(pseudoJ, residual);
      delta_params *= -1.0;                         
      
      return vectorToSTD(delta_params);
    }

    template<class T>
    StringDoubleMap computeStandardErrors(std::vector<T> &xdata_in,
                                          std::vector<double> &ydata_in,
                                          ModelFunction<T> model,
                                          CostFunction costFunction,
                                          double regularization,
                                          StringDoubleMap &paramsMap,
                                          StringDoubleMap &extraParameters){
      
      StringDoubleMap errors;
      
      vector residual = computeResiduals(xdata_in, ydata_in,
                                         model, costFunction,
                                         paramsMap, extraParameters);
      
      int nParams     = paramsMap.size();

      matrix J = computeJacobian(xdata_in, ydata_in, model,
                                 costFunction, paramsMap,
                                 extraParameters);
      
      
      matrix JT  = transpose(J);      
      matrix JTJ = matrixProduct(JT, J);
      
      for(int i = 0; i < nParams; ++i)
        JTJ(i, i) += regularization;
      
      
      matrix JTJ_inv     = inverse(JTJ);
      
      
      double residualSum = squaredNorm(residual);
      double variance    = residualSum / (ydata_in.size() - nParams);
      
      
      for (int i = 0; i < nParams; ++i) {
        errors[std::next(paramsMap.begin(), i)->first] = std::sqrt(variance * JTJ_inv(i, i));
      }
      return errors;
    }  
  }
}
