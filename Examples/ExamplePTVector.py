import FittingAlgorithms as fa
import numpy as np

# Polynomial model for vector data
def vector_model(xdata, params, extra_params):
    a = params["a"]
    b = params["b"]
    # xdata is a list of vectors, each with multiple dimensions
    return a * np.dot(xdata, xdata) + b * np.sum(xdata)

# Generate vector data: each point in xdata is a 3-dimensional vector
xdata = [x.tolist() for x in np.random.rand(20, 3) * 10]  # Convert each vector to a list

# True parameters used to generate the data
trueParameters = {"a": 2.5, "b": 5.2}

# Generate ydata based on the true parameters
ydata = [trueParameters["a"] * np.dot(x, x) + 
         trueParameters["b"] * np.sum(x) for x in xdata]

# Propose an initial guess for the parameters
initial_guesses = {"a": 1.0, "b": 1.0}



# Configure the Parallel-Tempering parameters
pt_params                = fa.ParallelTempering.Parameters()
pt_params.maxIterations  = 10000
pt_params.temperatures   = [1e-6, 1e-4, 1e-2, 1e0, 1e2]
pt_params.jumpSize       = 5*[0.001, 0.001, 0.001, 0.001, 0.001]
pt_params.numStepsSwap   = 1000
pt_params.numStepsFinish = 50000
pt_params.maxIterations  = 100000
pt_params.tolerance      = 1e-8
pt_params.printSteps     = 5000


# Run the fitting algorithm for vector data
parameters, error  = fa.ParallelTempering.fit(xdata, ydata, vector_model,
                                              initial_guesses,
                                              ptParams = pt_params)
# Display the results
print("Fitted Parameters:", parameters)
print("Target Parameters:", trueParameters)
