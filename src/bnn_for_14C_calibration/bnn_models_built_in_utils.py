# -*- coding: utf-8 -*-


import numpy as np

from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp

from .utils import (
    load_data
)
    
from typing import (
    Union, 
    Tuple, 
)

# ========================================================================
# lois a priori et a posteriori
# ========================================================================

def gaussian_prior(
    kernel_size: int,
    bias_size: int,
    dtype: tf.dtypes.DType = None,
    sigma: float = 1.0
) -> keras.Sequential:
    """
    Define a Gaussian prior distribution over weights and biases of a Bayesian neural network.

    Parameters
    ----------
    kernel_size : int
        Number of kernel weights in the layer.
    bias_size : int
        Number of bias terms in the layer.
    dtype : tf.dtypes.DType, optional
        TensorFlow data type for the distribution (default None).
    sigma : float, optional
        Standard deviation of the Gaussian prior (default 1.0).

    Returns
    -------
    keras.Sequential
        Non-trainable prior distribution model. Each weight and bias is assumed independent
        and distributed according to a normal distribution N(0, sigma^2).

    Notes
    -----
    - The prior distribution is **non-trainable**; its parameters are fixed.
    """
    n = kernel_size + bias_size
    var_mat_diag = sigma * tf.ones(n, dtype=dtype)
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n, dtype=dtype), scale_diag=var_mat_diag
                )
            )
        ]
    )
    return prior_model


def independent_gaussian_posterior(
    kernel_size: int,
    bias_size: int,
    dtype: tf.dtypes.DType = None
) -> keras.Sequential:
    """
    Define an independent Gaussian posterior distribution for variational inference
    in Bayesian neural networks.

    Parameters
    ----------
    kernel_size : int
        Number of kernel weights in the layer.
    bias_size : int
        Number of bias terms in the layer.
    dtype : tf.dtypes.DType, optional
        TensorFlow data type for the distribution (default None).

    Returns
    -------
    keras.Sequential
        Trainable posterior distribution model. Each weight and bias has a learnable mean
        and diagonal variance. Off-diagonal covariances are zero, implying independence
        among weights.

    Notes
    -----
    - The parameters of the distribution (mean and diagonal variance) are **trainable**.
    """
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.IndependentNormal.params_size(n), dtype=dtype
            ),
            tfp.layers.IndependentNormal(n),
        ]
    )
    return posterior_model

# ========================================================================
# fonction de perte
# ========================================================================

def negative_loglikelihood(
    targets: Union[np.ndarray, tf.Tensor],
    estimated_distribution: tfp.distributions.Distribution
) -> tf.Tensor:
    """
    Compute negative log-likelihood for a probabilistic prediction.

    Parameters
    ----------
    targets : np.ndarray or tf.Tensor
        True target values.
    estimated_distribution : tfp.distributions.Distribution
        Estimated (predicted) distribution output from the Bayesian network.

    Returns
    -------
    tf.Tensor
        Negative log-likelihood value for use as a loss function in Bayesian neural networks.

    Notes
    -----
    - Used with stochastic outputs of probabilistic networks.
    - Supports variational inference for posterior estimation.
    """
    return -estimated_distribution.log_prob(targets)

# ========================================================================
# fonctions d'aide pour les prédictions
# ========================================================================

def bnn_make_predictions_(
    bnn_model: keras.Model,
    X_test: Union[np.ndarray, tf.Tensor],
    iterations: int = 100,
    batch_size: int = None
) -> np.ndarray:
    """
    Generate predictions from a Bayesian neural network by repeated stochastic forward passes.

    Parameters
    ----------
    bnn_model : keras.Model
        Trained Bayesian neural network.
    X_test : np.ndarray or tf.Tensor
        Test input data.
    iterations : int, optional
        Number of stochastic forward passes to perform (default 100).
    batch_size : int, optional
        Batch size to use during prediction. If None, defaults to full batch.

    Returns
    -------
    np.ndarray
        Concatenated predictions from all iterations. Shape: (n_samples, n_outputs, iterations)

    Notes
    -----
    - Each call to the model produces a stochastic output due to the variational posterior.
    """
    predicted = []
    for _ in range(iterations):
        # !!! TO DO : investiger les différences de comportement entre 
        # model et model.predict avec batch_size != None !!!
        # predicted.append(bnn_model.predict(X_test, batch_size=batch_size, verbose=0))
        predicted.append(bnn_model(X_test))
    predicted = np.concatenate(predicted, axis=1)
    
    return predicted

# ========================================================================
# fonctions d'aide pour sauvegarder les prédictions
# ========================================================================

def bnn_load_predictions_(filepath: str) -> Tuple[np.ndarray, int, int]:
    """
    Load predictions generated by a Bayesian neural network from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing predictions.

    Returns
    -------
    predictions_array : np.ndarray
        Predictions as a float32 numpy array.
    nb_intervals : int
        Number of intervals (rows) in the predictions.
    nb_curves : int
        Number of curves (columns) in the predictions.

    Notes
    -----
    - Intended for predictions generated by `bnn_make_predictions_` for calibration purposes.
    """
    predictions_df = load_data(path=filepath, sep=",")
    predictions_array = predictions_df.to_numpy(dtype=np.float32)
    nb_intervals, nb_curves = predictions_array.shape
    return predictions_array, nb_intervals, nb_curves


# fonctions publiques du module
__all__ = [
    "gaussian_prior",
    "independent_gaussian_posterior",
    "bnn_make_predictions_",
    "bnn_load_predictions_"
]

# toutes les fonctions du module
all_functions = __all__
