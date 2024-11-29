#pragma once
#include <cmath>

namespace FittingAlgorithms {
    double squaredError(double ydata, double ypred);
    double relativeError(double ydata, double ypred);
    double squaredLogarimicError(double ydata, double ypred);
}
