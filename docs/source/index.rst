FittingAlgorithms documentation
===================================

Build
------------

.. code-block:: bash

    conda env create -f environment.yml
    conda activate fittingAlgorithms
    cmake -B build
    cmake --build build

Tests
------------
.. code-block:: bash

    ctest --output-on-failure --test-dir build


API Reference
==================

.. doxygennamespace:: FittingAlgorithms
   :project: FittingAlgorithms


Indices and tables
==================

   * :ref:`genindex`
   * :ref:`search`
