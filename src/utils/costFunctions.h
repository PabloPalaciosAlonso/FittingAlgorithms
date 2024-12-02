#pragma once
#include <cmath>

namespace FittingAlgorithms{
  
  inline double squaredError(double ytrue, double ypred){
    return (ytrue-ypred)*(ytrue-ypred);
  }

  inline double squaredRelativeError(double ytrue, double ypred){
    double relativeError = 1.0 - ypred/ytrue;
    return relativeError * relativeError;
  }
  
  inline double squaredLogarithmicError(double ytrue, double ypred){
    double logarithmicError = log(ytrue/ypred); 
    return logarithmicError * logarithmicError;
  }
}
