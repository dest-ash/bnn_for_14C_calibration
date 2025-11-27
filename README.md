# Bayesian Neural Network For Radiocarbon Calibration

Welcome to the GitHub repository of the **`bnn_for_14C_calibration`** Python library.

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
import numpy as np

######## plotting calibration curves ########

# the global BNN curve
bnn.plot_bnn_calibration_curve()

# the global BNN curve estimated using covariates 
# in addition, reset extra margins
bnn.plot_bnn_calibration_curve(covariables = True, reset_margins=True)

# only a portion of the global BNN curve estimated using covariates
bnn.plot_bnn_calibration_curve(
    covariables = True,
    Max_x=12400,
    Min_x=12200,
    Max_y=260,
    Min_y=180
)

# only the recent part of the BNN curve
bnn.plot_individual_calibration_curve_part_1()

# only the older part of the BNN curve
bnn.plot_individual_calibration_curve_part_2()

# IntCal20 curve
bnn.plot_IntCal20_curve(reset_margins=True)

######## individual calibration ########

# radiocarbon data
c14age = 10400
c14sig = 18

# independent calibration using BNN curve
individual_calib_results = bnn.individual_calibration(
    c14age, c14sig, 
    sample_size=10000,
    compute_calage_posterior_mean_and_std=True
)
print(individual_calib_results)

# independent calibration using the BNN curve trained with covariates
individual_calib_results_with_covariables = bnn.individual_calibration(
    c14age, c14sig, 
    sample_size=10000,
    covariables=True,
    compute_calage_posterior_mean_and_std=True
)
print(individual_calib_results_with_covariables)

# independent calibration using IntCal20 curve
IntCal20_calib_results = bnn.IntCal20_calibration(
    c14age, c14sig,
    sample_size=10000,
    compute_calage_posterior_mean_and_std=True
)
print(IntCal20_calib_results)

# plotting result for independent calibration

cm = 1/2.54  # cm in inches
figsize=(17*cm, 17*cm)

bnn.plot_calib_results(
    figsize=figsize,
    calibration_results=individual_calib_results
)

bnn.plot_calib_results(
    figsize=figsize,
    add_grid=True,
    color_cal_date='green',
    eps=-1,
    calibration_results=individual_calib_results_with_covariables
)

# same function for plotting IntCal20 calibration results
bnn.plot_calib_results(
    figsize=figsize,
    calibration_results=IntCal20_calib_results
)

######## joint calibration ########

c14ages=np.array([4000, 10483])
c14sigs=np.array([10, 18])

joint_calib_results = bnn.joint_calibration(
    c14ages = c14ages, 
    c14sigs = c14sigs,
    compute_calage_posterior_mode=True,
    chaine_length=10000
)
print(joint_calib_results)


######## local cache management ########

# test of cache downloading
bnn.download_cache_lib_data()

# force cache downloading
bnn.download_cache_lib_data(overwrite = True)

# clear cache
bnn.clear_cache()
```


## Reference

Refer to the pages on [**Calibration functions**](https://dest-ash.github.io/bnn_for_14C_calibration/reference/calibration), [**Plotting functions**](https://dest-ash.github.io/bnn_for_14C_calibration/reference/calib_plot_functions) and [**Local cache management**](https://dest-ash.github.io/bnn_for_14C_calibration/reference/manage_cache) in the **User Guide** of the [documentation](https://dest-ash.github.io/bnn_for_14C_calibration/) for detailed information on the inputs, outputs, and behaviour of each function.

For internal functions and more details on the implementation of all library functions, please refer to the **Developer's Guide** section of the [documentation](https://dest-ash.github.io/bnn_for_14C_calibration/).