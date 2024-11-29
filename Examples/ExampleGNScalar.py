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

# Optional configure the Gauss-Newton parameters
gn_params                = fa.GaussNewton.GNParameters()
gn_params.maxIterations  = 10000
gn_params.tolerance      = 1e-8
gn_params.printSteps     = 100
gn_params.regularization = 1e-5

#Propose an initial guess
initial_guesses = {"a": 1.0, "b": 1.0, "c":6}

# Run the fitting algorithm
parameters, errors = fa.GaussNewton.fit(xdata, ydata, model,
                                        initial_guesses,
                                        #costFunction = fa.squaredError, #Also can be a custom python function
                                        gnParams = gn_params)

print("Fitted Parameters:", parameters)
print("Target Parameters:", trueParameters)
