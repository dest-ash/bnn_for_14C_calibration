# -*- coding: utf-8 -*-


import numpy as np

from matplotlib import pyplot as plt
import pandas as pd


import warnings

from .utils import (
    get_lib_data_paths,
    minimax_scaling_reciproque, minimax_scaling,
    c14_to_f14c, c14sig_to_f14csig,
    #f14c_to_d14c, f14csig_to_d14csig,
    d14c_to_f14c, d14csig_to_f14csig,
    d14c_to_c14 #, d14csig_to_c14sig
)

from .bnn_models_built_in_utils import (
    bnn_make_predictions_, 
    bnn_load_predictions_
)

from .bnn_models_built_in import (
    bnn_load_model_part_2, 
    bnn_load_model_part_1,
    create_and_fit_Be10_curve, 
    create_and_fit_PaleoIntensity_curve, 
    create_features
)


from .calibration_utils import (
    _multi_cal_date_approx_density_,
    _mono_cal_date_approx_density_on_middle_points_, 
    compute_HPD_regions,
    mono_cal_date_approx_density_sample
)

from scipy.optimize import minimize


from typing import Dict, Any, Tuple, Optional
from typing import (
    # Literal,
    Optional,
    Union,
    # List,
    Tuple,
    # Callable,
    Dict,
    Any
)


# ========================================================================
# génération des chemins vers le cache local ou 
# les données embarquées dans le package
# ========================================================================

paths_results_dict = get_lib_data_paths()

# dossier contenant les données IntCal20
IntCal20_dir = paths_results_dict["IntCal20_dir"]

# dossier contenant les variables exogènes 
#covariates_dir = paths_results_dict["covariates_dir"]

# dossiers contenant les prédictions et les poids de modèles BNN
bnn_predictions_dir = paths_results_dict["bnn_predictions_dir"]
bnn_weights_dir = paths_results_dict["bnn_weights_dir"]


# =============================================================================
# calibration individuelle
# =============================================================================


def individual_calibration(
    c14age: float,
    c14sig: float,
    alpha: float = 0.05,
    covariables: bool = False,
    mesure_likelihood: str = [
        "gaussian_mixture", "curve_gaussian_approximation"
    ][0],
    compute_calage_posterior_mean_and_std: bool = False,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Perform an individual (independent) radiocarbon calibration using precomputed 
    BNN predictions.

    This function loads precomputed midpoint predictions for the two parts of the
    calibration curve (recent and older parts), constructs a posterior density on a
    common rescaled time axis, computes the Highest Posterior Density (HPD)
    region for the calibrated date, and returns a dictionary with results and
    diagnostics. Optionally, it can also compute Monte Carlo estimates of the
    posterior mean and standard deviation by sampling from the approximate
    posterior density.

    Parameters
    ----------
    c14age : float
        The measured radiocarbon age (conventional radiocarbon age) in years BP (Before Present).
    c14sig : float
        The 1-sigma measurement uncertainty of `c14age`.
    alpha : float, optional
        Tail probability for HPD region calculation (default 0.05).
    covariables : bool, optional
        If True, use models trained with covariates (Beryllium-10 and Earth's geomagnetic field 
        paleo-intensity); otherwise use models without covariates.
        Default is False.
    mesure_likelihood : str, optional, default="gaussian_mixture"
        Which measurement-likelihood model to use. Must be one of:
        "gaussian_mixture", "curve_gaussian_approximation".
        Default is "gaussian_mixture".
    compute_calage_posterior_mean_and_std : bool, optional
        If True, sample from the approximate posterior (via the piecewise approximation)
        and compute posterior mean and standard deviation. Default is False.
    sample_size : int, optional
        Number of posterior samples to draw when `compute_calage_posterior_mean_and_std` is True.
        Default is 1000.

    Returns
    -------
    results : Dict[str, Any]
        A dictionary containing many entries including (but not limited to):  
        - "calage_posterior_mode" : int  
            Posterior mode (most probable calibrated date, rounded to int).  
        - "calage_posterior_mode_density" : float  
            Posterior density (scaled) at the mode.  
        - "connexe_HPD_intervals" : ndarray  
            Array of connected HPD intervals in the *rescaled* domain.  
        - "connexe_HPD_intervals_unscaled" : ndarray  
            Same HPD intervals converted back to original date units.  
        - "connexe_HPD_intervals_unscaled_round" : ndarray  
            HPD intervals rounded to integers (useful as year ranges).  
        - "HPD_region_length" : int  
            Total length of HPD regions in original date units (rounded).  
        - "middle_points" : ndarray  
            Middle points (unscaled) used to compute the posterior density.  
        - "middle_points_density" : ndarray  
            Posterior density evaluated at `middle_points`.  
        - "calage_posterior_mean" : Optional[int]  
            Posterior mean (int) if sampling was requested, otherwise None.  
        - "calage_posterior_std" : Optional[int]  
            Posterior std (int) if sampling was requested, otherwise None.  
        - "calage_sample" : Optional[ndarray]  
            Posterior samples (in original date units) if sampling was requested, otherwise None.  
        - "c14age", "c14sig", "covariables", "alpha" : input parameters echoed back.  

    Notes
    -----
    **Mathematical formulation of measurement-likelihood model**

    The posterior density of the calibrated date $d$ (associated to a measurement $m$ of std 
    $\sigma$) is proportional to the product of the prior and the likelihood:
        $$
        p(d | m) \\propto p(d) \\times L(m | d),
        $$
    where the likelihood term \\( L(m | d) \\) is given by:
        $$
        L(m|d) = E_{BNN}[ \\exp(-(m - F^{14}C(d))^2 / (2σ^2)) ].
        $$

    Depending on the chosen `mesure_likelihood`, the likelihood term is estimated by:

    1. **Gaussian Mixture Approximation**
       $$
       \\hat{L}(m|d) = \\frac{1}{N} \\sum_{i=1}^N 
       \\frac{1}{\\sqrt{2\\pi}\\sigma} 
       \\exp\\left[-\\frac{(m - \\hat{F}^{14}_iC(d))^2}{2\\sigma^2}\\right]
       $$
       where \\( \\hat{F}^{14}_iC(d) \\) are stochastic predictions from the BNN.

    2. **Curve Gaussian Approximation (Central Limit Theorem)**
       $$
       \\hat{L}(m|d) = \\frac{1}{\\sqrt{2\\pi(\\sigma^2 + s_d^2)}} 
       \\exp\\left[-\\frac{(m - \\mu_d)^2}{2(\\sigma^2 + s_d^2)}\\right]
       $$
       where \\( \\mu_d \\) and \\( s_d \\) are the mean and std of BNN predictions.

    **Approximation and sampling strategy of the posterior distribution**

    The function relies on precomputed midpoint predictions loaded by
    `bnn_load_predictions_` and follows the piecewise approximation / sampling
    strategy used elsewhere in this package (as described for example in 
    `mono_cal_date_approx_density_sample`).

    Examples
    --------
    >>> ### TODO ###
    """
    
    # chargement des prédictions pré-sauvergardées
    
    if covariables :
        filename_part_2 = "bnn_part_2_with_covariables_middle_points_predictions.csv"
        filename_part_1 = "bnn_part_1_with_covariables_middle_points_predictions.csv"
    else :
        filename_part_2 = "bnn_part_2_without_covariables_middle_points_predictions.csv"
        filename_part_1 = "bnn_part_1_without_covariables_middle_points_predictions.csv"
    middle_points_predictions_part_2_filepath = bnn_predictions_dir / filename_part_2
    middle_points_predictions_part_1_filepath = bnn_predictions_dir / filename_part_1
        
    
    middle_points_predictions_part_2, nb_intervals_part_2, nb_curves_part_2 = bnn_load_predictions_(
        filepath = middle_points_predictions_part_2_filepath
    )
    
    middle_points_predictions_part_1, nb_intervals_part_1, nb_curves_part_1 = bnn_load_predictions_(
        filepath = middle_points_predictions_part_1_filepath
    )
    
    if nb_curves_part_1 != nb_curves_part_2 :
        # warnings.warn(
        #     f"""
        #     Le nombre de courbes utilisées dans pour la première partie est différent du nombre de courbes utilisées.
        #     nb_curves_part_1 = {nb_curves_part_1} tandis que nb_curves_part_2 = {nb_curves_part_2}.
        #     Il est préférable d'utiliser le même nombre de courbes pour avoir des estimateurs comparables lors de la
        #     calibration de nouvelles mesures.
        #     """,
        #     UserWarning,
        #     stacklevel=1
        # )
        # nb_curves = None
        raise ValueError(
            f"The number of predictive curves for the first and second parts differ: "
            f"nb_curves_part_1 = {nb_curves_part_1}, nb_curves_part_2 = {nb_curves_part_2}. "
            "You must use the same number of curves to obtain comparable estimators during calibration."
        )
    else :
        nb_curves = nb_curves_part_1
        
    if nb_intervals_part_1 != nb_intervals_part_2 :
        # warnings.warn(
        #     f"""
        #     Les subdivisions de l'intervalle de temps ne contiennent pas le même nombre d'intervalles sur les deux 
        #     parties de la courbe de calibration. En effet, nb_intervals_part_1 = {nb_intervals_part_1} et 
        #     nb_intervals_part_2 = {nb_intervals_part_2}. Il faut en tenir compte dans les algorithmes de calibration
        #     si ce n'est pas le cas.
        #     """,
        #     UserWarning,
        #     stacklevel=1
        # )
        warnings.warn(
            f"The time subdivisions do not contain the same number of intervals for the two parts of the calibration curve. "
            f"nb_intervals_part_1 = {nb_intervals_part_1} and nb_intervals_part_2 = {nb_intervals_part_2}. "
            "Please account for this in calibration algorithms used if needed.",
            UserWarning,
            stacklevel=1
        )
    
    # Paramétrages des bornes pour la partie ancienne de la courbe de calibration
    
    # Bornes des âges dans le dataset d'entraînement de la partie 2
    Max_part_2 = 63648
    Min_part_2 = 12283

    # choix bornes sup et inf adaptées à la partie part_2 (courbe partie ancienne) et leur reduction dans [0,1], 
    max_horizon_x_part_2 = 55000 # horizon temporel des ages calibrés limité à celui de IntCal20
    min_horizon_x_part_2 = 12310 # fin de la période avec dates incertaines
    max_horizon_x_part_2_scaled = minimax_scaling(max_horizon_x_part_2, Max=Max_part_2, Min=Min_part_2)
    min_horizon_x_part_2_scaled = minimax_scaling(min_horizon_x_part_2, Max=Max_part_2, Min=Min_part_2)

    # Subdivision de l'intervalle de temps couvert par la courbe part_2 en sous intervalles 
    # pour intégration et simulation si nécessaire
    intervals_bounds_part_2 = np.linspace(
        min_horizon_x_part_2_scaled, max_horizon_x_part_2_scaled, nb_intervals_part_2 + 1
    )
    middle_points_part_2 = (intervals_bounds_part_2[1:] + intervals_bounds_part_2[:-1])/2
    
    # Paramétrages des bornes pour la partie récente de la courbe de calibration
    
    # Bornes des âges dans le dataset d'entraînement de la partie 1
    Max_part_1 = 12310
    Min_part_1 = -4

    # choix bornes sup et inf adaptées à la partie part_1 (courbe partie récente) et leur reduction dans [0,1], 
    max_horizon_x_part_1 = 12310 # limite des dates absolument connus dans les données d'entraînement disponibles
    min_horizon_x_part_1 = -4 # la date la plus récente dans le jeu de données
    max_horizon_x_part_1_scaled = minimax_scaling(max_horizon_x_part_1, Max=Max_part_1, Min=Min_part_1)
    min_horizon_x_part_1_scaled = minimax_scaling(min_horizon_x_part_1, Max=Max_part_1, Min=Min_part_1)

    # Subdivision de l'intervalle de temps couvert par la courbe part_1 en sous intervalles 
    # pour intégration et simulation si nécessaire
    intervals_bounds_part_1 = np.linspace(
        min_horizon_x_part_1_scaled, max_horizon_x_part_1_scaled, nb_intervals_part_1 + 1
    )
    middle_points_part_1 = (intervals_bounds_part_1[1:] + intervals_bounds_part_1[:-1])/2
    
    # fonction  de densité de la date pour cette mesure aux points milieu de la subdivision de l'intervalle

    density_on_middle_points = _mono_cal_date_approx_density_on_middle_points_(
        mesure = c14_to_f14c(c14=c14age), 
        lab_error = c14sig_to_f14csig(c14=c14age, c14sig=c14sig),
        middle_points_predictions= np.concatenate(
            [
                # premier tableau : prédictions partie 1
                d14c_to_f14c(
                    d14c = middle_points_predictions_part_1,
                    teta = minimax_scaling_reciproque(
                        x = middle_points_part_1,
                        Max = Max_part_1,
                        Min = Min_part_1
                    ).repeat(nb_curves).reshape((-1,nb_curves))
                ),
                
                # second tableau : prédictions partie 2
                d14c_to_f14c(
                    d14c = middle_points_predictions_part_2,
                    teta = minimax_scaling_reciproque(
                        x = middle_points_part_2,
                        Max = Max_part_2,
                        Min = Min_part_2
                    ).repeat(nb_curves).reshape((-1,nb_curves))
                )
            ],
            axis=0
        ),
        mesure_likelihood = mesure_likelihood,
        prior_density = "default"
    )
    
    # évaluation de la densité aux points milieu
    
    # on fait d'abord un nouveau rescale des points milieux
    # pour qu'ils soient contenus dans [0,1]
    # avec 0 correspondant à la date min de part_1
    # et 1 à la date max de part_2
    middle_points_part_1_unscaled = minimax_scaling_reciproque(
        middle_points_part_1, 
        Max=Max_part_1, 
        Min=Min_part_1
    )
    middle_points_part_2_unscaled = minimax_scaling_reciproque(
        middle_points_part_2, 
        Max=Max_part_2, 
        Min=Min_part_2
    )
    middle_points_rescaled = np.concatenate(
        [
            minimax_scaling(
                middle_points_part_1_unscaled,
                Max=Max_part_2,
                Min=Min_part_1
            ),
            minimax_scaling(
                middle_points_part_2_unscaled,
                Max=Max_part_2,
                Min=Min_part_1
            )
        ]
    )
    
    # calcul de la densité correspondant aux points rescaled
    middle_points_density = density_on_middle_points(middle_points_rescaled)
    
    # calcul de la région HPD 
    
    # on fait d'abord un nouveau rescale des bornes de sous-intervalles
    # pour qu'ils soient contenus dans [0,1]
    # avec 0 correspondant à la date min de part_1
    # et 1 à la date max de part_2
    intervals_bounds_part_1_unscaled = minimax_scaling_reciproque(
        intervals_bounds_part_1, 
        Max=Max_part_1, 
        Min=Min_part_1
    )
    intervals_bounds_part_2_unscaled = minimax_scaling_reciproque(
        intervals_bounds_part_2, 
        Max=Max_part_2, 
        Min=Min_part_2
    )
    intervals_bounds_rescaled = np.concatenate(
        [
            minimax_scaling(
                intervals_bounds_part_1_unscaled,
                Max=Max_part_2,
                Min=Min_part_1
            ),
            minimax_scaling(
                intervals_bounds_part_2_unscaled,
                Max=Max_part_2,
                Min=Min_part_1
            )
        ]
    )
    
    # finalement, calcul des régions HPD correspondant à cette subdivision rescaled
    HPD_regions_results = compute_HPD_regions(
        alpha = alpha, 
        subdivision_components=(intervals_bounds_rescaled, middle_points_rescaled, middle_points_density)
    )
    
    # if HPD_regions_results['calage_posterior_mode'] > intervals_bounds_part_1[-1] :
    #     HPD_regions_results['calage_posterior_mode'] = int(minimax_scaling_reciproque(
    #         x=HPD_regions_results['calage_posterior_mode'], 
    #         Max=Max_part_2, 
    #         Min=Min_part_2
    #     ))
    # else :
    #     HPD_regions_results['calage_posterior_mode'] = int(minimax_scaling_reciproque(
    #         x=HPD_regions_results['calage_posterior_mode'], 
    #         Max=Max_part_1, 
    #         Min=Min_part_1
    #     ))
        
    HPD_regions_results['calage_posterior_mode'] = int(minimax_scaling_reciproque(
            x=HPD_regions_results['calage_posterior_mode'], 
            Max=Max_part_2, 
            Min=Min_part_1
        ))

    HPD_regions_results['connexe_HPD_intervals'] = np.array(HPD_regions_results['connexe_HPD_intervals'])
    HPD_regions_results['connexe_HPD_intervals_unscaled'] = minimax_scaling_reciproque(
        x=HPD_regions_results['connexe_HPD_intervals'], 
        Max=Max_part_2, 
        Min=Min_part_1
    )
    HPD_regions_results['connexe_HPD_intervals_unscaled_round'] = np.array([
        [int(interval[0]), int(interval[1])+1] for interval in HPD_regions_results['connexe_HPD_intervals_unscaled']
    ])

    HPD_regions_results['HPD_region_length'] = 0
    connexe_HPD_intervals_unscaled = HPD_regions_results['connexe_HPD_intervals_unscaled']
    nb_HPDI = connexe_HPD_intervals_unscaled.shape[0]
    for i in range(nb_HPDI) :
        HPD_regions_results['HPD_region_length'] += connexe_HPD_intervals_unscaled[i,1] - connexe_HPD_intervals_unscaled[i,0]
    HPD_regions_results['HPD_region_length'] = int(HPD_regions_results['HPD_region_length'])+1
    
    results: Dict[str, Any] = HPD_regions_results
    
    results['alpha'] = alpha
    
    results['middle_points'] = minimax_scaling_reciproque(
        x = middle_points_rescaled,
        Max = Max_part_2,
        Min = Min_part_1
    )
    results['middle_points_density'] = middle_points_density
    
    if compute_calage_posterior_mean_and_std :
        calage_sample, _, _ = mono_cal_date_approx_density_sample(
             subdivision_components = (intervals_bounds_rescaled, middle_points_rescaled, middle_points_density), 
             sample_size = sample_size
        )
        calage_sample = minimax_scaling_reciproque(
            x=calage_sample, 
            Max=Max_part_2, 
            Min=Min_part_1
        )
        calage_mean = calage_sample.mean()
        calage_std = calage_sample.std()
        results['calage_posterior_mean'] = int(calage_mean)
        results['calage_posterior_std'] = int(calage_std) + 1
        results['calage_sample'] = calage_sample
    else :
        results['calage_posterior_mean'] = None
        results['calage_posterior_std'] = None
        results['calage_sample'] = None

    results['c14age'] = c14age
    results['c14sig'] = c14sig
    results['covariables'] = covariables
    
    return results



def IntCal20_calibration(
    c14age: float,
    c14sig: float,
    alpha: float = 0.05,
    compute_calage_posterior_mean_and_std: bool = False,
    sample_size: int = 1000
) -> Dict[str, Any]:
    """
    Perform Bayesian calibration of a radiocarbon measurement using the IntCal20 calibration curve.

    Parameters
    ----------
    c14age : float
        Radiocarbon measured age (in BP).
    c14sig : float
        Laboratory uncertainty associated with the radiocarbon age (standard deviation, in BP).
    alpha : float, optional (default = 0.05)
        Significance level used to compute Highest Posterior Density (HPD) region.
        The returned region covers posterior probability mass equal to (1 - alpha).
    compute_calage_posterior_mean_and_std : bool, optional (default = False)
        If True, posterior samples of the calibrated age are drawn and used to compute:  
        - posterior mean,  
        - posterior standard deviation.  
        If False, the sampling step is skipped.
    sample_size : int, optional (default = 1000)
        Number of posterior samples to draw when
        `compute_calage_posterior_mean_and_std = True`.

    Returns
    -------
    results : Dict[str, Any]
        A dictionary containing:
        
        - `connexe_HPD_intervals` : array of HPD intervals in scaled space  
        - `connexe_HPD_intervals_unscaled` : HPD intervals transformed back into calendar years  
        - `connexe_HPD_intervals_unscaled_round` : integer-rounded calendar HPD intervals  
        - `calage_posterior_mode` : posterior mode of the calibrated date (calendar scale)  
        - `HPD_region_length` : total length (in years) of the HPD region  
        - `middle_points` : calendar ages at which posterior density was evaluated  
        - `middle_points_density` : posterior density evaluated at these points  
        - `calage_posterior_mean` : posterior mean (if computed)  
        - `calage_posterior_std` : posterior standard deviation (if computed)  
        - `calage_sample` : posterior sample values (if sampling was performed)  
        - `alpha` : significance level used  
        - `c14age`, `c14sig` : the original radiocarbon measurement  
        - `covariables` : always None for IntCal20 curve

    Notes
    -----
    **Calibration model**

    The posterior distribution of the calendar date `d` is:
    $$
        p(d | m) \\propto p(d) \\times L(m | d),
    $$
    where:  
        - $m$ is the measured radiocarbon age of std $\sigma$ (in the F14C domain),  
        - $L(m | d)$ is the measurement-likelihood (Gaussian error model in the F14C domain),  
        - $p(d)$ is the prior (here uniform over the range covered by IntCal20 calibration curve).  
    When calibration is done with IntCal20, the likelihood term is estimated by
    $$
       \\hat{L}(m|d) = \\frac{1}{\\sqrt{2\\pi(\\sigma^2 + s_{IntCal20}(d)^2)}} 
       \\exp\\left[-\\frac{(m - \\mu_{IntCal20}(d))^2}{2(\\sigma^2 + s_{IntCal20}(d)^2)}\\right]
    $$
    where \\( \\mu_{IntCal20}(d) \\) and \\( s_{IntCal20}(d)\\) are the mean and std published for 
    the IntCal20 curve (they are interpolated for points where they are not given). This likelihood
    estimate comes from a Gaussian approximation of the curve obtained by Central Limit Theorem.

    **Approximation and sampling strategy of the posterior distribution**

    The interval covered by IntCal20 is divided into a large number of
    small intervals (typically one every ≈2 years), and on each interval
    the posterior density is approximated by piecewise-constant values.
    This allows:

    - fast HPD region extraction,
    - deriving a sampling strategy.

    Further description about this can be found in the documentation of
    `mono_cal_date_approx_density_sample` for example.

    **HPD computation**

    HPD regions are obtained by sorting discretized posterior density values
    and finding the smallest set of intervals that accumulates posterior mass
    `(1 - alpha)`.

    **Optional posterior moments**

    If `compute_calage_posterior_mean_and_std=True`:

    - posterior samples are drawn using piecewise-constant inverse transform sampling,  
    - calendar ages are recovered by inverse scaling,  
    - sample mean and variance are computed.

    Examples
    --------
    >>> out = IntCal20_calibration(2850, 30)
    >>> out['connexe_HPD_intervals_unscaled_round']
    array([[ 900, 1020]])

    >>> # Computing posterior mean and standard deviation
    >>> out = IntCal20_calibration(2850, 30, compute_calage_posterior_mean_and_std=True)
    >>> out['calage_posterior_mean']
    960
    >>> out['calage_posterior_std']
    42
    """

    # courbe IntCal20
    IntCal20_file_path = IntCal20_dir / "IntCal20_completed.csv"
    IntCal20 = pd.read_csv(IntCal20_file_path, sep =",")
    
    # récupération du nombre de points milieu 
    nb_intervals = 30000 # environ 1 point tous les deux ans sur l'intervalle de 0 à 55000
    
    # Paramétrages des bornes
    
    # Bornes des âges dans le dataset
    Max_part_2 = 55000
    Min_part_2 = 0 #12310

    # choix bornes sup et inf adaptées à la partie part_2 (courbe partie ancienne) et leur reduction dans [0,1], 
    max_horizon_x = 55000 # horizon temporel des ages calibrés limité à celui de IntCal20
    min_horizon_x_part_2 = 0 #12310 # fin de la période avec dates incertaines
    max_horizon_x_part_2_scaled = minimax_scaling(max_horizon_x, Max=Max_part_2, Min=Min_part_2)
    min_horizon_x_part_2_scaled = minimax_scaling(min_horizon_x_part_2, Max=Max_part_2, Min=Min_part_2)

    # Subdivision de l'intervalle de temps couvert par la courbe part_2 en sous intervalles 
    # pour intégration et simulation si nécessaire
    intervals_bounds_part_2 = np.linspace(min_horizon_x_part_2_scaled, max_horizon_x_part_2_scaled, nb_intervals + 1)
    middle_points_part_2 = (intervals_bounds_part_2[1:] + intervals_bounds_part_2[:-1])/2
    
    # courbe IntCal20 interpolée aux points milieu
    teta = minimax_scaling_reciproque(
        x = middle_points_part_2,
        Max = Max_part_2,
        Min = Min_part_2
    )
    moyenne = np.interp(
        x = teta,
        xp = IntCal20.calage.values[::-1], 
        fp = IntCal20.d14c.values[::-1]
    )
    ecart_type = np.interp(
        x = teta,
        xp = IntCal20.calage.values[::-1], 
        fp = IntCal20.d14csigma.values[::-1]
    )
    middle_points_predictions_part_2 = np.array(
        [
            # moyenne dans le domaine F14C
            d14c_to_f14c(
                d14c = moyenne,
                teta = teta
            ),
            
            # écart-type dans le domaine F14C
            d14csig_to_f14csig(
                d14csig = ecart_type,
                teta = teta
            )
        ]
    )
    
    
    # fonction  de densité de la date pour cette mesure aux points milieu de la subdivision de l'intervalle

    density_on_middle_points = _mono_cal_date_approx_density_on_middle_points_(
        mesure = c14_to_f14c(c14=c14age), 
        lab_error = c14sig_to_f14csig(c14=c14age, c14sig=c14sig),
        middle_points_predictions= middle_points_predictions_part_2,
        mesure_likelihood = 'IntCal20',
        prior_density = "default"
    )
    
    # évaluation de la densité aux points milieu

    middle_points_density = density_on_middle_points(middle_points_part_2)
    
    # calcul de la région HPD 
    
    HPD_regions_results = compute_HPD_regions(
        alpha = alpha, 
        subdivision_components=(intervals_bounds_part_2, middle_points_part_2, middle_points_density)
    )
    
    HPD_regions_results['calage_posterior_mode'] = int(minimax_scaling_reciproque(
        x=HPD_regions_results['calage_posterior_mode'], 
        Max=Max_part_2, 
        Min=Min_part_2
    ))

    HPD_regions_results['connexe_HPD_intervals'] = np.array(HPD_regions_results['connexe_HPD_intervals'])
    HPD_regions_results['connexe_HPD_intervals_unscaled'] = minimax_scaling_reciproque(
        x=HPD_regions_results['connexe_HPD_intervals'], 
        Max=Max_part_2, 
        Min=Min_part_2
    )
    HPD_regions_results['connexe_HPD_intervals_unscaled_round'] = np.array([
        [int(interval[0]), int(interval[1])+1] for interval in HPD_regions_results['connexe_HPD_intervals_unscaled']
    ])

    HPD_regions_results['HPD_region_length'] = 0
    connexe_HPD_intervals_unscaled = HPD_regions_results['connexe_HPD_intervals_unscaled']
    nb_HPDI = connexe_HPD_intervals_unscaled.shape[0]
    for i in range(nb_HPDI) :
        HPD_regions_results['HPD_region_length'] += connexe_HPD_intervals_unscaled[i,1] - connexe_HPD_intervals_unscaled[i,0]
    HPD_regions_results['HPD_region_length'] = int(HPD_regions_results['HPD_region_length'])+1
    
    results = HPD_regions_results
    
    results['alpha'] = alpha
    
    results['middle_points'] = minimax_scaling_reciproque(
        x = middle_points_part_2,
        Max = Max_part_2,
        Min = Min_part_2
    )
    results['middle_points_density'] = middle_points_density
    
    if compute_calage_posterior_mean_and_std :
        calage_sample, _, _ = mono_cal_date_approx_density_sample(
             subdivision_components = (intervals_bounds_part_2, middle_points_part_2, middle_points_density), 
             sample_size = sample_size
        )
        calage_sample = minimax_scaling_reciproque(
            x=calage_sample, 
            Max=Max_part_2, 
            Min=Min_part_2
        )
        calage_mean = calage_sample.mean()
        calage_std = calage_sample.std()
        results['calage_posterior_mean'] = int(calage_mean)
        results['calage_posterior_std'] = int(calage_std) + 1
        results['calage_sample'] = calage_sample
    else :
        results['calage_posterior_mean'] = None
        results['calage_posterior_std'] = None
        results['calage_sample'] = None

    results['c14age'] = c14age
    results['c14sig'] = c14sig
    results['covariables'] = None
    
    return results



# =============================================================================
# calibration jointe
# =============================================================================


# fonction concanténant les valeurs issues de deux parties de la courbe
def concatenate_curves_parts(
    covariables: bool = False
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Concatenate the two parts of the radiocarbon calibration curve
    (recent part and ancient part) by loading pre-computed midpoint predictions,
    rescaling the domains, and merging both sets into a single calibration structure.

    The function performs the following steps:

    1. Load midpoint predictions for the two curve parts from previously saved files.  
    2. Verify that both parts were generated using the same number of curves.
    3. Optionally warn the user if the two parts do not share the same number of
       temporal subdivisions.
    4. Recover original time scales using inverse min-max scaling and convert midpoint
       predictions from Δ14C (d14c) to F14C values.
    5. Rescale both curve parts into a unified [0, 1] domain so that 
        0 corresponds to the minimum of the recent part (part 1) and
        1 corresponds to the maximum of the ancient part (part 2).
    6. Return all processed quantities as a structured dictionary that is
       directly usable for individual calibration, density evaluation, and
       HPD interval computation.

    Parameters
    ----------
    covariables : bool, optional
        Whether covariables were used when producing the Bayesian Neural Network
        midpoint predictions. Defaults to False.

    Returns
    -------
    Dict[str, Union[int, float, numpy.ndarray]]:
        A dictionary containing calibration domain information and concatenated
        prediction data, with the following keys:

        **Curve part 1 (recent period)** 

        - ``'Max_part_1'`` (float): Maximum value of the calendar dates in training data for part 1.  
        - ``'Min_part_1'`` (float): Minimum value of the calendar dates in training data for part 1.  
        - ``'max_horizon_x_part_1'`` (float): Upper limit of the calibration domain (unscaled) for part 1.  
        - ``'max_horizon_x_part_1_scaled'`` (float): Corresponding upper bound after min–max scaling.  
        - ``'min_horizon_x_part_1'`` (float): Lower limit of the calibration domain (unscaled) for part 1.  
        - ``'min_horizon_x_part_1_scaled'`` (float): Corresponding lower bound after min–max scaling.  

        **Curve part 2 (ancient period)**  

        - ``'Max_part_2'`` (float): Maximum value of the calendar dates in training data for part 2.
        - ``'Min_part_2'`` (float): Minimum value of the calendar dates in training data for part 2.
        - ``'max_horizon_x_part_2'`` (float): Upper limit of the calibration domain (unscaled) for part 2.
        - ``'max_horizon_x_part_2_scaled'`` (float): Corresponding upper bound after min–max scaling.
        - ``'min_horizon_x_part_2'`` (float): Lower limit of the calibration domain (unscaled) for part 2.
        - ``'min_horizon_x_part_2_scaled'`` (float): Corresponding lower bound after min–max scaling.

        **Concatenated and processed quantities**  

        - ``'nb_curves'`` (int): Number of neural network models simultaneously used to produce predictions.
        - ``'intervals_bounds_rescaled'`` (numpy.ndarray): Rescaled boundaries of all time subdivisions after merging.
        - ``'middle_points_rescaled'`` (numpy.ndarray): Midpoints of all subdivisions after conversion into [0, 1].
        - ``'middle_points_predictions_in_F14C'`` (numpy.ndarray):
          Matrix ``(nb_intervals_total × nb_curves)`` containing midpoint predictions
          expressed as F14C values after reconstruction of the original time scale.

    Raises
    ------
    ValueError
        If the two curve parts were generated using different numbers of curves.

    Warns
    -----
    UserWarning
        If the two curve parts do not share the same number of time subdivisions.

    Notes
    -----
    The merging and rescaling ensure that sampling and density computations in
    individual calibration operate in a consistent 1D standardized domain while
    preserving the physical meaning of the original temporal scales.
    """
    
    # chargement des prédictions pré-sauvergardées
    
    if covariables :
        filename_part_2 = "bnn_part_2_with_covariables_middle_points_predictions.csv"
        filename_part_1 = "bnn_part_1_with_covariables_middle_points_predictions.csv"
    else :
        filename_part_2 = "bnn_part_2_without_covariables_middle_points_predictions.csv"
        filename_part_1 = "bnn_part_1_without_covariables_middle_points_predictions.csv"
    middle_points_predictions_part_2_filepath = bnn_predictions_dir / filename_part_2
    middle_points_predictions_part_1_filepath = bnn_predictions_dir / filename_part_1
        
    
    middle_points_predictions_part_2, nb_intervals_part_2, nb_curves_part_2 = bnn_load_predictions_(
        filepath = middle_points_predictions_part_2_filepath
    )
    
    middle_points_predictions_part_1, nb_intervals_part_1, nb_curves_part_1 = bnn_load_predictions_(
        filepath = middle_points_predictions_part_1_filepath
    )
    
    if nb_curves_part_1 != nb_curves_part_2 :
        raise ValueError(
            f"""
            Le nombre de courbes utilisées dans pour la première partie est différent du nombre de courbes utilisées.
            nb_curves_part_1 = {nb_curves_part_1} tandis que nb_curves_part_2 = {nb_curves_part_2}.
            Il est nécessaire d'utiliser le même nombre de courbes pour avoir des estimateurs comparables lors de la
            calibration de nouvelles mesures.
            """
        )
    else :
        nb_curves = nb_curves_part_1
        
    if nb_intervals_part_1 != nb_intervals_part_2 :
        warnings.warn(
            f"""
            Les subdivisions de l'intervalle de temps ne contiennent pas le même nombre d'intervalles sur les deux 
            parties de la courbe de calibration. En effet, nb_intervals_part_1 = {nb_intervals_part_1} et 
            nb_intervals_part_2 = {nb_intervals_part_2}. Il faut en tenir compte dans les algorithmes de calibration
            si ce n'est pas le cas.
            """,
            UserWarning,
            stacklevel=1
        )
    
    # Paramétrages des bornes pour la partie ancienne de la courbe de calibration
    
    # Bornes des âges dans le dataset d'entraînement de la partie 2
    Max_part_2 = 63648
    Min_part_2 = 12283

    # choix bornes sup et inf adaptées à la partie part_2 (courbe partie ancienne) et leur reduction dans [0,1], 
    max_horizon_x_part_2 = 55000 # horizon temporel des ages calibrés limité à celui de IntCal20
    min_horizon_x_part_2 = 12310 # fin de la période avec dates incertaines
    max_horizon_x_part_2_scaled = minimax_scaling(max_horizon_x_part_2, Max=Max_part_2, Min=Min_part_2)
    min_horizon_x_part_2_scaled = minimax_scaling(min_horizon_x_part_2, Max=Max_part_2, Min=Min_part_2)

    # Subdivision de l'intervalle de temps couvert par la courbe part_2 en sous intervalles 
    # pour intégration et simulation si nécessaire
    intervals_bounds_part_2 = np.linspace(
        min_horizon_x_part_2_scaled, max_horizon_x_part_2_scaled, nb_intervals_part_2 + 1
    )
    middle_points_part_2 = (intervals_bounds_part_2[1:] + intervals_bounds_part_2[:-1])/2
    
    # Paramétrages des bornes pour la partie récente de la courbe de calibration
    
    # Bornes des âges dans le dataset d'entraînement de la partie 1
    Max_part_1 = 12310
    Min_part_1 = -4

    # choix bornes sup et inf adaptées à la partie part_1 (courbe partie récente) et leur reduction dans [0,1], 
    max_horizon_x_part_1 = 12310 # limite des dates absolument connus dans les données d'entraînement disponibles
    min_horizon_x_part_1 = -4 # la date la plus récente dans le jeu de données
    max_horizon_x_part_1_scaled = minimax_scaling(max_horizon_x_part_1, Max=Max_part_1, Min=Min_part_1)
    min_horizon_x_part_1_scaled = minimax_scaling(min_horizon_x_part_1, Max=Max_part_1, Min=Min_part_1)

    # Subdivision de l'intervalle de temps couvert par la courbe part_1 en sous intervalles 
    # pour intégration et simulation si nécessaire
    intervals_bounds_part_1 = np.linspace(
        min_horizon_x_part_1_scaled, max_horizon_x_part_1_scaled, nb_intervals_part_1 + 1
    )
    middle_points_part_1 = (intervals_bounds_part_1[1:] + intervals_bounds_part_1[:-1])/2
    
    
    # conversion des prédictions aux points milieu dans le domaine
    # F14C et concanténation des deux parties de la courbe
    middle_points_predictions_in_F14C = np.concatenate(
        [
            # premier tableau : prédictions partie 1
            d14c_to_f14c(
                d14c = middle_points_predictions_part_1,
                teta = minimax_scaling_reciproque(
                    x = middle_points_part_1,
                    Max = Max_part_1,
                    Min = Min_part_1
                ).repeat(nb_curves).reshape((-1,nb_curves))
            ),

            # second tableau : prédictions partie 2
            d14c_to_f14c(
                d14c = middle_points_predictions_part_2,
                teta = minimax_scaling_reciproque(
                    x = middle_points_part_2,
                    Max = Max_part_2,
                    Min = Min_part_2
                ).repeat(nb_curves).reshape((-1,nb_curves))
            )
        ],
        axis=0
    )
    
    
    # on fait un nouveau rescale des points milieux
    # pour qu'ils soient contenus dans [0,1]
    # avec 0 correspondant à la date min de part_1
    # et 1 à la date max de part_2
    # (utile pour le calcul de la densité en calibration individuelle
    # et pour la simulation suivant cette densité)
    middle_points_part_1_unscaled = minimax_scaling_reciproque(
        middle_points_part_1, 
        Max=Max_part_1, 
        Min=Min_part_1
    )
    middle_points_part_2_unscaled = minimax_scaling_reciproque(
        middle_points_part_2, 
        Max=Max_part_2, 
        Min=Min_part_2
    )
    middle_points_rescaled = np.concatenate(
        [
            minimax_scaling(
                middle_points_part_1_unscaled,
                Max=Max_part_2,
                Min=Min_part_1
            ),
            minimax_scaling(
                middle_points_part_2_unscaled,
                Max=Max_part_2,
                Min=Min_part_1
            )
        ]
    )
    
    
    
    # on fait un nouveau rescale des bornes de sous-intervalles
    # pour qu'ils soient contenus dans [0,1]
    # avec 0 correspondant à la date min de part_1
    # et 1 à la date max de part_2
    # (utile pour le calcul des régions HPD en calibration individuelle
    # et pour la simulation suivant la densité associée)
    intervals_bounds_part_1_unscaled = minimax_scaling_reciproque(
        intervals_bounds_part_1, 
        Max=Max_part_1, 
        Min=Min_part_1
    )
    intervals_bounds_part_2_unscaled = minimax_scaling_reciproque(
        intervals_bounds_part_2, 
        Max=Max_part_2, 
        Min=Min_part_2
    )
    intervals_bounds_rescaled = np.concatenate(
        [
            minimax_scaling(
                intervals_bounds_part_1_unscaled,
                Max=Max_part_2,
                Min=Min_part_1
            ),
            minimax_scaling(
                intervals_bounds_part_2_unscaled,
                Max=Max_part_2,
                Min=Min_part_1
            )
        ]
    )
    
    
    # retour des résultats à utiliser pendant la calibration
    return {
        # part 1
        'Max_part_1': Max_part_1,
        'Min_part_1': Min_part_1,
        'max_horizon_x_part_1': max_horizon_x_part_1,
        'max_horizon_x_part_1_scaled': max_horizon_x_part_1_scaled,
        'min_horizon_x_part_1': min_horizon_x_part_1,
        'min_horizon_x_part_1_scaled': min_horizon_x_part_1_scaled,
        
        # part 2
        'Max_part_2': Max_part_2,
        'Min_part_2': Min_part_2,
        'max_horizon_x_part_2': max_horizon_x_part_2,
        'max_horizon_x_part_2_scaled': max_horizon_x_part_2_scaled,
        'min_horizon_x_part_2': min_horizon_x_part_2,
        'min_horizon_x_part_2_scaled': min_horizon_x_part_2_scaled,
        
        # concatenated results
        'nb_curves': nb_curves,
        'intervals_bounds_rescaled': intervals_bounds_rescaled,
        'middle_points_rescaled': middle_points_rescaled,
        'middle_points_predictions_in_F14C': middle_points_predictions_in_F14C
    }



# MCMC pour calibration jointe
def multi_cal_date_approx_density_MCMC_sampler_for_concatenated_curve(
    mesures, 
    lab_errors,
    
    covariables = False,
    prior_density="default",
    marginal_prior_density="default",
    #ordered = False,
    #adapt_proposals = True,
    chaine_length = 100,
    # batch_size pour les prédictions des proposals en mode minibatch : 
    # par défaut None, ce qui utilise chaine_length (soit dim_chaine steps)
    batch_size = None
) :
    
    # chargement des modèles entrainés
    bnn_model_part_1 = bnn_load_model_part_1(
        path_to_model_weigths='last_version',
        covariables=covariables
    )
    
    bnn_model_part_2 = bnn_load_model_part_2(
        path_to_model_weigths='last_version',
        covariables=covariables
    )
    
    # résultats issus de la concatenation de deux parties de la courbe
    concatenated_results = concatenate_curves_parts(
        covariables = covariables
    )
    
    middle_points_predictions = concatenated_results['middle_points_predictions_in_F14C']
    middle_points = concatenated_results['middle_points_rescaled']
    intervals_bounds = concatenated_results['intervals_bounds_rescaled']
    nb_curves = concatenated_results['nb_curves']
    
    Max_part_1 = concatenated_results['Max_part_1']
    Min_part_1 = concatenated_results['Min_part_1']
    #max_horizon_x_part_1 = concatenated_results['max_horizon_x_part_1']
    #max_horizon_x_part_1_scaled = concatenated_results['max_horizon_x_part_1_scaled']
    #min_horizon_x_part_1 = concatenated_results['min_horizon_x_part_1']
    #min_horizon_x_part_1_scaled = concatenated_results['min_horizon_x_part_1_scaled']
    
    Max_part_2 = concatenated_results['Max_part_2']
    Min_part_2 = concatenated_results['Min_part_2']
    #max_horizon_x_part_2 = concatenated_results['max_horizon_x_part_2']
    #max_horizon_x_part_2_scaled = concatenated_results['max_horizon_x_part_2_scaled']
    min_horizon_x_part_2 = concatenated_results['min_horizon_x_part_2']
    #min_horizon_x_part_2_scaled = concatenated_results['min_horizon_x_part_2_scaled']
    
    
    # création et instanciation des modèles de génération des covariables si requis
    if covariables :
        # partie de la courbe sans dates incertaines
        covariables_list_models_part_1=[
            # modèle de paléo-intensité
            create_and_fit_PaleoIntensity_curve(
                Max_age=Max_part_1, 
                Min_age=Min_part_1
            ),
            
            # modèle de Béryllium 10
            create_and_fit_Be10_curve(
                Max_age=Max_part_1,
                Min_age=Min_part_1,
                alpha=1e0, 
                n_knots=100
            )
        ]
        
        
        # partie de la courbe avec dates incertaines
        covariables_list_models_part_2=[
            # modèle de paléo-intensité
            create_and_fit_PaleoIntensity_curve(
                Max_age=Max_part_2, 
                Min_age=Min_part_2
            ),
            
            # modèle de Béryllium 10
            create_and_fit_Be10_curve(
                Max_age=Max_part_2,
                Min_age=Min_part_2,
                alpha=1e0, 
                n_knots=100
            )
        ]
        
        # valeurs max et min obtenues pour les covariables générées lors
        # de la phase d'entraînement des réseaux chargés précédemment
        # (les max et min sont les mêmes pour les deux parties de la courbe)
        # on a des listes de la forme : 
        # [val max ou min pour paleo-intensité, val max ou min pour Be10]
        covariables_max_values_from_training_stage = [11.138879098711064, 1.3907171663086664]
        covariables_min_values_from_training_stage = [7.418579938520401, 0.8666379944417881]
    
    
    # dimension de la chaîne = nombre de mesures à calibrer
    dim_chaine = mesures.shape[0] # = len(mesures)
    
    # initialisation des proposals unidimensionnelles et leurs probabilités
    proposals = np.empty(shape = (dim_chaine, chaine_length))
    proposals_proba = np.empty((dim_chaine, chaine_length))
    for j in range(dim_chaine) :
        # densité marginale dim j
        if type(middle_points_predictions) != np.ndarray :
            raise ValueError(
                """
                'middle_points_predictions' doit être de type 'numpy.ndarray'
                quand il est fourni et de taille ('nb_intervals', 'nb_curves')
                """
            )
        marginal_density_dim_j = _mono_cal_date_approx_density_on_middle_points_(
            mesure = mesures[j], 
            lab_error = lab_errors[j],
            # à la place du bnn_model, on utilise plutot les prédictions fournies par 
            # middle_points_predictions qui sont déjà exprimées dans le domaine F14C
            middle_points_predictions = middle_points_predictions, 
            prior_density = marginal_prior_density
        )

        # évaluation de la densité aux points milieu
        middle_points_density_j = marginal_density_dim_j(middle_points)

        # tirage des chaine_length proposals suivant cette densité et 
        # récupération de la proba (non normalisée) associée à chaque observation tirée
        proposals_dim_j, unscaled_proba_dim_j, _ = mono_cal_date_approx_density_sample(
             subdivision_components = (intervals_bounds, middle_points, middle_points_density_j), 
             sample_size = chaine_length
        )

        # sauvegarde des propasals et de leurs probas
        # N.B : les proposals sont dans [0,1] ici 
        # avec 0 correspondant à Min_part_1 et 1 à Max_part_2
        # il faudra corriger avant de faire les prédictions car :
        # bnn_model_part_1 recoit des proposals dont 0 doit renvoyer à Min_part_1 et 1 à Max_part_1, et
        # bnn_model_part_2 recoit des proposals dont 0 doit renvoyer à Min_part_2 et 1 à Max_part_2
        proposals[j,] = proposals_dim_j
        proposals_proba[j,] = unscaled_proba_dim_j
        
    # on met les probas à échelle logarithmique
    proposals_proba = np.log(proposals_proba)
    
    # calcul des prédictions aux différents proposals pour utilisation ultérieure dans
    # le calcul de la densité jointe cible (afin de ne pas prédire chaque fois dans le sampler)
    
    # echelle de temps des âges calibrés
    proposals_unscaled = minimax_scaling_reciproque(
        x=proposals,
        Max=Max_part_2,
        Min=Min_part_1
    )
    
    # les indices des dates relevant de la partie de la courbe avec dates absolues
    idx_part_1 = np.where(
        proposals_unscaled.reshape((-1,1)) < min_horizon_x_part_2
    )[0]
    
    # les indices des dates relevant de la partie de la courbe avec dates incertaines
    idx_part_2 = np.where(
        proposals_unscaled.reshape((-1,1)) >= min_horizon_x_part_2
    )[0]
    
    if batch_size == None :
        batch_size = chaine_length 
        # pas utilisé pour l'instant dans bnn_make_predictions_ à cause des différences entre l'évaluation
        # de model comparée à celle de model.predict avec batch_size != None.
        # on garde donc l'évaluation de model qui fait la prédiction sur le dataset en entrée tout entier
        # sans mode minibatch. (les prédictions model.predict sont comme "bruitées")
        # pas d'énormes différences en terme de performance en temps d'exécution entre les deux modes de 
        # prédiction sur la taille des chaines que nous utilisons jusqu'à présent (jusqu'à 10000 par exemple)
        
    # initialisation des prédictions
    proposals_predictions = np.empty(
        shape = (dim_chaine * chaine_length, nb_curves)
    )
    
    # calcul des prédictions relevant du premier modèle (partie de la courbe sans dates incertaines)
    if len(idx_part_1) > 0 :
        predictors = minimax_scaling(
            x=proposals_unscaled.reshape((-1,1))[idx_part_1,],
            Max=Max_part_1,
            Min=Min_part_1
        )
        if covariables:
            predictors = create_features(
                predictors, 
                covariables_max_values_from_training_stage = covariables_max_values_from_training_stage, 
                covariables_min_values_from_training_stage = covariables_min_values_from_training_stage,
                covariables_list_models=covariables_list_models_part_1
            )[0]
        
        proposals_predictions[idx_part_1,] = bnn_make_predictions_(
            bnn_model = bnn_model_part_1, 
            X_test = predictors, 
            iterations = nb_curves, 
            batch_size = batch_size
        )
    
    # calcul des prédictions relevant du second modèle (partie de la courbe avec dates incertaines)
    if len(idx_part_2) > 0 :
        predictors = minimax_scaling(
                x=proposals_unscaled.reshape((-1,1))[idx_part_2,],
                Max=Max_part_2,
                Min=Min_part_2
            )
        if covariables:
            predictors = create_features(
                predictors, 
                covariables_max_values_from_training_stage = covariables_max_values_from_training_stage, 
                covariables_min_values_from_training_stage = covariables_min_values_from_training_stage,
                covariables_list_models=covariables_list_models_part_2
            )[0]
        proposals_predictions[idx_part_2,] = bnn_make_predictions_(
            bnn_model = bnn_model_part_2, 
            X_test = predictors, 
            iterations = nb_curves, 
            batch_size = batch_size
        )
    
    # on met les predictions dans le domaine F14C (domaine de calibration) :
    
    # celles issues du premier modèle :
    if len(idx_part_1) > 0 :
        proposals_predictions[idx_part_1,] = d14c_to_f14c(
            d14c = proposals_predictions[idx_part_1,],
            teta = (
                proposals_unscaled.reshape((-1,1))[idx_part_1,]
            ).repeat(nb_curves).reshape(-1,nb_curves)
        )
        
    # celles issues du second modèle :
    if len(idx_part_2) > 0 :
        proposals_predictions[idx_part_2,] = d14c_to_f14c(
            d14c = proposals_predictions[idx_part_2,],
            teta = (
                proposals_unscaled.reshape((-1,1))[idx_part_2,]
            ).repeat(nb_curves).reshape(-1,nb_curves)
        )
    
    # mettre les prédictions des proposals sous format (dim_chaine, chaine_length, nb_curves)
    proposals_predictions = proposals_predictions.reshape((dim_chaine, chaine_length, nb_curves))
    
    # choix point de départ de la chaine : point dont les coordonées sont les proposals de plus grande probabilité :
    
    
    rows_index = range(dim_chaine)
    
    # ici résultats issus éventuellement de plusieurs colonnes différentes
    ind_proposals = proposals_proba.argmax(axis=1)
    
    chaine_init_state = np.copy(proposals[rows_index,ind_proposals])
    chaine_last_accepted_proposals_marginal_proba = np.copy(proposals_proba[rows_index,ind_proposals])
    chaine_last_accepted_proposals_predictions = np.copy(proposals_predictions[rows_index,ind_proposals,:])
        
    # initialisation de la chaîne
    chaine = np.empty((dim_chaine,chaine_length))
    chaine[:,0] = chaine_init_state
    
    # création de la densité jointe (cible) sur les dates et évaluation au point courant de la chaîne
    target_joint_density = _multi_cal_date_approx_density_(
        #ordered = ordered,
        mesures = mesures,
        lab_errors = lab_errors,
        nb_curves = nb_curves
    )
    
    # sauvegarde de la log-densité jointe de la chaine 
    # (densité jointe connue à une constante près)
    chaine_log_density = np.empty((chaine_length,))
    
    # variable pour tracker la log-densité jointe du dernier état de la chaîne
    chaine_last_accepted_state_log_density = np.log(target_joint_density(
        np.array([np.copy(chaine[:,0])]),
        chaine_last_accepted_proposals_predictions.reshape((1,dim_chaine,nb_curves))
    ))[0]
    
    # log-densité jointe du point initial de la chaine
    chaine_log_density[0] = chaine_last_accepted_state_log_density
    
    # systematic scan M.-H. within Gibbs sampler
    rng = np.random.default_rng()
    marginal_acceptance_rates = np.zeros(dim_chaine)
    global_acceptance_rate = 0
    for n in range(1,chaine_length) : 
        chaine[:,n] = np.copy(chaine[:,n-1])
        for j in range(dim_chaine) :
            proposal_n_dim_j = proposals[j,n]
            unscaled_proba_proposal_n_dim_j = proposals_proba[j,n]
    
            proposal_state_n_next_value = np.copy(chaine[:,n])
            proposal_state_n_next_value[j] = proposal_n_dim_j
    
            proposal_prediction_state_n_next_value = np.copy(chaine_last_accepted_proposals_predictions)
            proposal_prediction_state_n_next_value[j,:] = np.copy(proposals_predictions[j,n,:])
    
            log_joint_proba_on_current_and_proposal_states = np.log(target_joint_density(
                np.array([np.copy(chaine[:,n]), proposal_state_n_next_value]),
                np.concatenate((
                    chaine_last_accepted_proposals_predictions.reshape((1,dim_chaine,nb_curves)),
                    proposal_prediction_state_n_next_value.reshape((1,dim_chaine,nb_curves))
                ))
            ))
    
            acceptance_proba = np.exp(
                log_joint_proba_on_current_and_proposal_states[1] +
                chaine_last_accepted_proposals_marginal_proba[j] -
                log_joint_proba_on_current_and_proposal_states[0] -
                unscaled_proba_proposal_n_dim_j
            )
    
            u = rng.random() # loi uniforme sur [0,1]
            if u <= acceptance_proba :
                chaine[j,n] = proposal_n_dim_j
                chaine_last_accepted_proposals_marginal_proba[j] = unscaled_proba_proposal_n_dim_j
                chaine_last_accepted_proposals_predictions[j,:] = np.copy(proposals_predictions[j,n,:])
                marginal_acceptance_rates[j] = marginal_acceptance_rates[j] + 1
                
                chaine_last_accepted_state_log_density = log_joint_proba_on_current_and_proposal_states[1]
    
        global_acceptance_rate = global_acceptance_rate + (chaine[:,n] != chaine[:,n-1]).prod()
        
        chaine_log_density[n] = chaine_last_accepted_state_log_density
        
    
    # changement de toutes les coordonnées
    global_acceptance_rate = global_acceptance_rate / (chaine_length - 1)
    
    # changement d'au moins une coordonnée
    modified_global_acceptance_rate = np.any((chaine[:,:-1] != chaine[:,1:]),axis=0).mean() 
    
    # changements pour chaque coordonnée
    marginal_acceptance_rates = marginal_acceptance_rates / (chaine_length - 1)
    
    # conversion de la chaîne vers l'échelle des âges calibrés (dates calendaires)
    chaine = minimax_scaling_reciproque(
        x=chaine,
        Max=Max_part_2,
        Min=Min_part_1
    )
        
    return {
        'chaine': chaine, 
        'chaine_log_density': chaine_log_density, 
        'global_acceptance_rate': global_acceptance_rate,
        'modified_global_acceptance_rate': modified_global_acceptance_rate, 
        'marginal_acceptance_rates': marginal_acceptance_rates
    }



def joint_calibration(
    c14ages, c14sigs,
    alpha = 0.05, # 1-0.68, #0.05,
    #ordered=False,
    covariables=False,
    #batch_size=None,
    compute_calage_posterior_mean_and_std = True,
    compute_calage_posterior_mode = False,
    chaine_length = 1000 # à utiliser pour la taille de la chaine MCMC
) :
    
    # passage du domaine C14 au domaine F14C
    c14ages = c14_to_f14c(c14=c14ages)
    c14sigs = c14sig_to_f14csig(c14=c14ages, c14sig=c14sigs)
    
    # MCMC outputs
    MCMC_results = multi_cal_date_approx_density_MCMC_sampler_for_concatenated_curve(
        covariables = covariables,
        #ordered = ordered,
        #batch_size=batch_size,
        mesures = c14ages, 
        lab_errors = c14sigs,
        chaine_length=chaine_length
    )
    
    
    chaine = MCMC_results['chaine']
    chaine_log_density = MCMC_results['chaine_log_density']
    acceptance_rate = MCMC_results['modified_global_acceptance_rate']
    marginal_acceptance_rates = MCMC_results['marginal_acceptance_rates']
    
    results = {
        # si oui (cf. argument ordered), relation croissante supposée sur les dates calibrées
        #'ordre_sur_dates' : ordered,
        'chaine' : chaine, 
        'chaine_log_joint_density_unscaled' : chaine_log_density,
        'acceptance_rate' : acceptance_rate,
        'marginal_acceptance_rates' : marginal_acceptance_rates
    }
    
    
    if compute_calage_posterior_mode :
        mode_index = chaine_log_density.argmax()
        results['mode_index'] = mode_index
        results['calage_posterior_mode'] = np.int64(chaine[:,mode_index])
    else :
        results['mode_index'] = None
        results['calage_posterior_mode'] = None
    
    if compute_calage_posterior_mean_and_std :
        calage_mean = np.int64(chaine.mean(axis=1))
        calage_std = np.int64(chaine.std(axis=1))+1
        results['calage_posterior_mean'] = calage_mean
        results['calage_posterior_std'] = calage_std
    else :
        results['calage_posterior_mean'] = None
        results['calage_posterior_std'] = None
    
    return results



__all__ = [
    # fonctions de calibration
    "individual_calibration", 
    "IntCal20_calibration",
    "concatenate_curves_parts",
    "multi_cal_date_approx_density_MCMC_sampler_for_concatenated_curve", 
    "joint_calibration"
]