# Bayesian Neural Network For Radiocarbon Calibration

Welcome to the official documentation for the **`bnn_for_14C_calibration`** Python library.

## Overview

This library implements algorithms based on Bayesian Neural Networks (BNN) for $^{14}$C calibration.  
Among available functionalities, we can find : the independent calibration of a $^{14}$C age (individual calibration), the joint calibration of several $^{14}$C ages (simultaneous calibration), the independent calibration using the IntCal20 curve for comparisons with BNN methods results or the plotting of calibration results and curves.

## Installation

### Python version and installation

This library **requires** `Python 3.9.13` version to work. 

If you do not have Python installed on your computer, you can install the [**Anaconda** distribution](https://www.anaconda.com/docs/getting-started/anaconda/install) (or [**Miniconda**](https://www.anaconda.com/docs/getting-started/miniconda/install) for a lightweight distribution).

Once the installation is complete, open a terminal (on Linux or macOS) or Anaconda Prompt (on Windows) and create a conda environment with the required Python version (`3.9.13`) and pip (the Python package manager) as follows:

```bash
conda create --name myenv python=3.9.13 pip
```

`myenv` is the name of of your environment.The command above creates an environement called `myenv` in which it installs `pip` and `Python 3.9.13`.

To learn more about managing Python environments with conda, please visit [this page](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Installing `bnn_for_14C_calibration`

Before installing `bnn_for_14C_calibration`, you need to activate your conda environment as follows:

```bash
conda activate myenv
```

Then, the library can be installed using `pip` as follows:

```bash
pip install bnn_for_14C_calibration
```

If you wish, you can also install `jupyter-notebook` in your environment (`myenv`) to test the library:

```bash
pip install jupyter
```

You can then launch the notebook using the `jupyter-notebook` command in the terminal and conduct your experiments there.

## Quick Start

Example usage : 

```python
import bnn_for_14C_calibration as bnn

# plotting calibration curves

# independent calibration
result = main_function(data)
print(result)

# plotting result for independent calibration

# joint calibration
```


## API Reference

See the API documentation for detailed information.