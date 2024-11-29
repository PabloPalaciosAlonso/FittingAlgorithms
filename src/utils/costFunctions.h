#pragma once
#include <cmath>

namespace FittingAlgorithms {
  double squaredError(double ydata, double ypred);
  double squaredRelativeError(double ydata, double ypred);
  double squaredLogarithmicError(double ydata, double ypred);
}
