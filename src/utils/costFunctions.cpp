#pragma once
#include<iostream>
#include <cmath>
#include"costFunctions.h"

namespace FittingAlgorithms{
  
  double squaredError(double ytrue, double ypred){
    return (ytrue-ypred)*(ytrue-ypred);
  }

  double squaredRelativeError(double ytrue, double ypred){
    double relativeError = 1.0 - ypred/ytrue;
    return relativeError * relativeError;
  }
  
  double squaredLogarithmicError(double ytrue, double ypred){
    double logarithmicError = log(ytrue/ypred); 
    return logarithmicError * logarithmicError;
  }
}
