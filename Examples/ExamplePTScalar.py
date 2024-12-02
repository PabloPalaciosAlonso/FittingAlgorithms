import FittingAlgorithms as fa
import numpy as np

#Polynomic model
def model(xdata, params, extra_params):
    a = params["a"]
    b = params["b"]
    c = params["c"]    
    return a * xdata**2 + b*xdata + c


#Generate the data using the target parameters
trueParameters = {"a":2.5, "b":1.2, "c":3.33}
xdata          = np.linspace(0,10)
ydata          = model(xdata, trueParameters, {})

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


#Propose an initial guess
initial_guesses = 5*[{"a": 1.0, "b": 10.0, "c":6}]

# Run the fitting algorithm
parameters, error = fa.ParallelTempering.fit(xdata, ydata, model,
                                             initial_guesses,
                                             #costFunction = fa.squaredError, #Also can be a custom python function
                                             ptParams = pt_params)

print("Fitted Parameters:", parameters)
print("Target Parameters:", trueParameters)

