#include <GaussNewton.h>
#include <ParallelTempering.h>
#include <costFunctions.h>
#include <defines.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using FittingAlgorithms::StringDoubleMap;

// Damped exponential with a linear bias and fixed baseline.
double dampedLineModel(double x, const StringDoubleMap &params,
                       const StringDoubleMap &extra) {
  const double amplitude = params.at("amplitude");
  const double decay = params.at("decay");
  const double bias = params.at("bias");
  const double baseline = extra.at("baseline");
  return baseline + amplitude * std::exp(-decay * x) + bias * x;
}

void printParameters(const StringDoubleMap &params, const std::string &title) {
  std::cout << title << "\n";
  for (const auto &entry : params) {
    std::cout << "  " << std::setw(10) << entry.first << ": " << entry.second
              << "\n";
  }
}

int main() {
  const StringDoubleMap trueParameters{
      {"amplitude", 4.2}, {"decay", 0.35}, {"bias", -0.07}};
  const StringDoubleMap extraParameters{{"baseline", 1.5}};

  std::vector<double> xdata;
  std::vector<double> ydata;
  xdata.reserve(80);
  ydata.reserve(80);

  std::mt19937 rng(42);
  std::normal_distribution<double> noise(0.0, 0.03);

  for (int i = 0; i < 80; ++i) {
    double x = 0.1 * i;
    double clean = dampedLineModel(x, trueParameters, extraParameters);
    xdata.push_back(x);
    ydata.push_back(clean + noise(rng));
  }

  std::cout << "Fitting with Gauss-Newton and Parallel Tempering\n\n";

  FittingAlgorithms::GaussNewton::Parameters gnParams;
  gnParams.maxIterations = 500;
  gnParams.tolerance = 1e-7;
  gnParams.printSteps = 100;
  gnParams.regularization = 1e-5;

  StringDoubleMap gnInitial{
      {"amplitude", 3.0}, {"decay", 0.15}, {"bias", 0.05}};

  auto gnX = xdata;
  auto gnY = ydata;
  auto gnResult = FittingAlgorithms::GaussNewton::fit<double>(
      gnX, gnY, dampedLineModel, gnInitial, gnParams,
      FittingAlgorithms::squaredRelativeError, extraParameters);

  printParameters(trueParameters, "Target parameters");
  printParameters(gnResult.parameters, "Gauss-Newton fit");
  printParameters(gnResult.errors, "Gauss-Newton std errors");

  std::cout << "\nRunning Parallel Tempering search...\n";

  std::vector<double> temperatures{1e-4, 1e-3, 1e-2, 1e-1, 1.0};
  std::vector<double> jumpSizes(temperatures.size(), 0.02);

  std::vector<StringDoubleMap> ptInitial;
  ptInitial.reserve(temperatures.size());
  for (size_t i = 0; i < temperatures.size(); ++i) {
    ptInitial.push_back({{"amplitude", gnInitial.at("amplitude") + noise(rng)},
                         {"decay", gnInitial.at("decay") + 0.05 * noise(rng)},
                         {"bias", gnInitial.at("bias") + 0.1 * noise(rng)}});
  }

  FittingAlgorithms::ParallelTempering::Parameters ptParams;
  ptParams.temperatures = temperatures;
  ptParams.jumpSize = jumpSizes;
  ptParams.numStepsSwap = 200;
  ptParams.numStepsFinish = 4000;
  ptParams.maxIterations = 20000;
  ptParams.tolerance = 1e-6;
  ptParams.printSteps = 1000;
  ptParams.numThreads = 4;

  auto ptX = xdata;
  auto ptY = ydata;
  auto ptResult = FittingAlgorithms::ParallelTempering::fit<double, double, StringDoubleMap>(ptX, ptY, dampedLineModel, ptInitial, ptParams,
                                                                                             FittingAlgorithms::squaredLogarithmicError, extraParameters);

  printParameters(ptResult, "Parallel Tempering fit");

  return 0;
}
