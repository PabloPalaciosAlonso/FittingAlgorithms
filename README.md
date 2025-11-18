# FittingAlgorithms

A C++ project implementing algorithms for fitting data to model.
## Usage

### Install dependencies 

This project uses [conda](https://github.com/conda-forge/miniforge) to manage dependencies. Install conda if you don't have it already, then create and activate the environment:

```bash
conda env create
conda activate fittingAlgorithms
```

### Build

Standard CMake build workflow:

```bash
cmake -B build
cmake --build build
```

Binaries will be generated under the `build` directory.

## Run tests

Unit tests are located in the `Tests` folder. After building the project, you can run the tests as follows:

```bash
cd build
ctest
```
