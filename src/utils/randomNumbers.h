#pragma once
#include<random>

namespace FittingAlgorithms {
  std::mt19937& getRng() {
    static std::random_device rd;
    static std::mt19937 rng(rd());
    return rng;
  }
  
  int generateRandomInt(int first, int last) {
    std::uniform_int_distribution<int> dist(first, last);
    return dist(getRng());
  }
  
  double generateRandomDouble(double first, double last) {
    std::uniform_real_distribution<double> dist(first, last);
    return dist(getRng());
  }
}
