#pragma once
#include<iostream>
#include <cmath>
#include"costFunctions.h"

namespace FittingAlgorithms{
  
  double squaredError(double ytrue, double ypred){
    return (ytrue-ypred)*(ytrue-ypred);
  }

  double relativeError(double ytrue, double ypred){
    return fabs((ytrue-ypred))/ytrue;
  }

  double squaredLogarimicError(double ytrue, double ypred){
    return log(ytrue/ypred)*log(ytrue/ypred);
  }
}
