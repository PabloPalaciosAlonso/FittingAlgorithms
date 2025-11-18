#pragma once
#include <iostream>
#include "utils/defines.h"

namespace FittingAlgorithms{
  template <typename T>
  inline ModelFunction<T> createModelFunction(py::function model) {
    return [model](const T& x,
                   const std::map<std::string, double>& params,
                   const std::map<std::string, double>& extraParams) -> double {
      py::object result = model(x, params, extraParams);
      return result.cast<double>();
    };
  }
  
  inline CostFunction createCostFunction(py::function costFunction) {
    return [costFunction](double ydata, double ypred) -> double {
      return costFunction(ydata, ypred).cast<double>();
    };
  }
}
