#pragma once
#include <cmath>
#include <stdexcept>


namespace FittingAlgorithms{

  /**
   * @brief Computes the squared error between a target value and a predicted value.
   *
   * @param ytarget Target value.
   * @param ypred Predicted value.
   * @return The squared error \f$(y_{\text{target}} - y_{\text{pred}})^2\f$.
   */
  inline double squaredError(double ytarget, double ypred){
    return (ytarget-ypred)*(ytarget-ypred);
  }

  /**
   * @brief Computes the squared relative error between a target and a predicted value.
   *
   * @param ytarget Target value.
   * @param ypred Predicted value.
   * @return The squared relative error \f$(1 - y_{\text{pred}}/y_{\text{target}})^2\f$.
   */
  inline double squaredRelativeError(double ytarget, double ypred){
    if (ytarget == 0.0)
      throw std::runtime_error("squaredRelativeError: ytarget cannot be zero.");
    double relativeError = 1.0 - ypred/ytarget;
    return relativeError * relativeError;
  }

  /**
   * @brief Computes the squared logarithmic error between a target and a predicted value.
   * @param ytarget Target value.
   * @param ypred Predicted value.
   * @return The squared logarithmic error
   *         \f$(\log(y_{\text{target}} / y_{\text{pred}}))^2\f$.
   */
  inline double squaredLogarithmicError(double ytarget, double ypred){
    if (ytarget*ypred <= 0.0)
      throw std::runtime_error("squaredLogarithmicError: ytarget and ypred must have the same sign.");
    double logarithmicError = log(ytarget/ypred); 
    return logarithmicError * logarithmicError;
  }
}
