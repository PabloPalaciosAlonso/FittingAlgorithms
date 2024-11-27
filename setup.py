from setuptools import setup, Extension
import pybind11
import os

# Ruta al directorio base del proyecto
base_dir = os.path.abspath(os.path.dirname(__file__))

# Incluir directorios de encabezados
include_dirs = [
    pybind11.get_include(),  # Incluir PyBind11
    os.path.join(base_dir, "src"),  # Incluir tu código fuente
    os.path.join(base_dir, "wrappers"),   # Encabezados adicionales (si los hay)
    
]

# Configurar la extensión
extensions = [
    Extension(
        name="FittingAlgorithms",  # Nombre del módulo principal
        sources=[
            "wrappers/FittingAlgorithmsWrapper.cpp",  # Wrapper del módulo principal
            "wrappers/GaussNewtonWrapper.cpp",  # Código fuente de Gauss-Newton
            "src/GaussNewton.cpp",  # Código fuente de Gauss-Newton
        ],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=["-std=c++17"],  # Configuración del compilador
    )
]

# Configurar la instalación
setup(
    name="FittingAlgorithms",
    version="1.0.0",
    author="Tu Nombre",
    author_email="tu.email@example.com",
    description="Módulo Python para algoritmos de ajuste, incluyendo Gauss-Newton.",
    long_description="Bindings de Python para varios algoritmos de ajuste implementados en C++.",
    ext_modules=extensions,
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        "pybind11>=2.6.0",  # Dependencia de PyBind11
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
