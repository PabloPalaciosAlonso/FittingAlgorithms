import FittingAlgorithms as fa
import numpy as np

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
gn_params = fa.GaussNewton.GNParameters()
gn_params.maxIterations = 100
gn_params.tolerance = 1e-6
gn_params.printSteps = 10

# Ajusta los parámetros
result = fa.GaussNewton.fit(xdata, ydata, gn_params, model, initial_guesses, {})

print("Parameters:", result.parameters)
print("Errors:", result.errors)


# Modelo adaptado para vectores
def vector_model(xdata, ydata, params, extra_params):
    a = params["a"]
    b = params["b"]
    c = params["c"]
    
    # xdata es ahora una lista de vectores, cada uno con múltiples dimensiones
    return [a * np.dot(x, x) + b * np.sum(x) + c - y for x, y in zip(xdata, ydata)]

def vector_func(x, params, extra_params):
    a = params["a"]
    b = params["b"]
    c = params["c"]
    
    # x es un vector
    return a * np.dot(x, x) + b * np.sum(x) + c

# Define los datos y parámetros

# Generamos datos vectoriales: cada punto en xdata es un vector de dimensión 3
xdata = [np.random.rand(3) * 10 for _ in range(20)]
realParameters = {"a": 2.5, "b": 2.2, "c": 1}

# Calculamos ydata para cada vector en xdata
ydata = [vector_func(x, realParameters, {}) for x in xdata]

initial_guesses = {"a": 1.0, "b": 1.0, "c": 1.0}

# Configura los parámetros del algoritmo
gn_params = fa.GaussNewton.GNParameters()
gn_params.maxIterations = 100
gn_params.tolerance = 1e-6
gn_params.printSteps = 10

# Ajusta los parámetros usando el modelo vectorial
result = fa.GaussNewton.fit(xdata, ydata, gn_params, vector_model, initial_guesses, {})

# Imprime los resultados
print("Parameters:", result.parameters)
print("Errors:", result.errors)
