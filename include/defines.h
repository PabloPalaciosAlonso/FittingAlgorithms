#pragma once
#include <iostream>
#include <map>
#include <functional>

namespace FittingAlgorithms{
  using StringDoubleMap = std::map<std::string, double>;
  using CostFunction    = std::function<double(const double, const double)>;
  using ModelFunction   = std::function<double(const double,
                                               const StringDoubleMap&,
                                               const StringDoubleMap&)>;
}
