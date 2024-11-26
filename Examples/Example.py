import gauss_newton
import numpy as np


# Define el modelo en Python
def model(xdata, ydata, params, extra_params):
    a = params["a"]
    b = params["b"]
    c = params["c"]
    
    return [a * x**2 + b*x +c - y for x,y in zip(xdata, ydata)]

def func(x, params, extra_params):
    a = params["a"]
    b = params["b"]
    c = params["c"]

    return a*x**2 + b*x + c
    
    

# Define los datos y parámetros

realParameters = {"a":2.5, "b":1.2, "c":-3}

xdata = np.linspace(0,10)
ydata = func(xdata, realParameters, {})

initial_guesses = {"a": 1.0, "b": 1.0, "c":1}

# Configura los parámetros del algoritmo
gn_params = gauss_newton.GNParameters()
gn_params.maxIterations = 100
gn_params.tolerance = 1e-6
gn_params.printSteps = 10

# Ajusta los parámetros
result = gauss_newton.fitParams(xdata, ydata, gn_params, model, initial_guesses, {})

print("Parameters:", result.parameters)
print("Errors:", result.errors)
