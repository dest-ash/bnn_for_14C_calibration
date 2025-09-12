# -*- coding: utf-8 -*-


import numpy as np

from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp

from .utils import (
    load_data
)
    


# ========================================================================
# lois a priori et a posteriori
# ========================================================================


def gaussian_prior(kernel_size, bias_size, dtype=None, sigma=1):
    """
    
    Parameters
    ----------
    kernel_size : TYPE int
        DESCRIPTION.
    bias_size : TYPE int
        DESCRIPTION.
    dtype : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    prior_model : TYPE distribution "non entrainable"
        DESCRIPTION.  Définit la loi à priori pour chaque poids (et biais) du réseau comme une loi normale de moyenne = 0 et écart type = 1.
        On peut noter que la distribution a priori n'est pas entraînable ici, car ses paramètres sont fixés

    """
    n = kernel_size + bias_size
    var_mat_diag = sigma * tf.ones(n)
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=var_mat_diag
                )
            )
        ]
    )
    return prior_model


def independent_gaussian_posterior(kernel_size, bias_size, dtype=None) :
    """

    Parameters
    ----------
    kernel_size : TYPE int
        DESCRIPTION.
    bias_size : TYPE int
        DESCRIPTION.
    dtype : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    posterior_model : TYPE "distribution entrainable"
        DESCRIPTION. Définit la distribution variationnelle de poids à postériori comme une gaussienne multivariée indépendante.
        On peut noter que les paramètres apprenables pour cette distribution sont la moyenne et la matrice diagonale variances-covariances
        (les covariances sont nulles, ce qui implique que les poids sont indépendants pour une gaussienne)

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

def negative_loglikelihood(targets, estimated_distribution):
    """

    Parameters
    ----------
    targets : TYPE float
        DESCRIPTION. les valeurs de la variable cible
    estimated_distribution : TYPE "distribution"
        DESCRIPTION. la vraissemble ("estimée") du (par le) modèle

    Returns
    -------
    TYPE function
        DESCRIPTION. l'opposé de la log-vraissemblance comme fonction de perte à utiliser dans le cadre des réseaux bayésiens probabilistes 
        (sortie du réseau stochastique/aléatoire) lors de l'estimation de la loi à postériori par inférence variationnelle

    """
    return -estimated_distribution.log_prob(targets)

# ========================================================================
# fonctions d'aide pour les prédctions
# ========================================================================


# à utiliser durant la phase de calibration
def bnn_make_predictions_(bnn_model, X_test, iterations=100, batch_size = None) :
    predicted = []
    for _ in range(iterations):
        # !!! TO DO : investiger les différences de comportement entre 
        # model et model.predict avec batch_size != None !!!
        # predicted.append(bnn_model.predict(X_test, batch_size=batch_size, verbose=0))
        predicted.append(bnn_model(X_test))
    predicted = np.concatenate(predicted, axis=1)
    
    return predicted

# ========================================================================
# fonctions d'aide pour sauvegarder les prédctions
# ========================================================================


# pour télécharger les prédictions à utiliser en calibration 
# données par la fonction bnn_make_predictions_
def bnn_load_predictions_(filepath):
    predictions_df = load_data(path = filepath, sep = ",")
    predictions_array = predictions_df.to_numpy(dtype=np.float32)
    nb_intervals, nb_curves = predictions_array.shape
    return predictions_array, nb_intervals, nb_curves


    