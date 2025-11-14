# -*- coding: utf-8 -*-


import numpy as np
from scipy.optimize import minimize

from typing import (
    # Literal,
    Optional,
    Union,
    List,
    Tuple,
    Callable
)

# =============================================================================
# approximation de la densité a posteriori des dates calibrées
# =============================================================================

# version calibration individuelle d'une seule date

def mono_cal_date_approx_density(
    mesure: float,
    lab_error: float,
    bnn_model: object,
    nb_curves: int = 100,
    prior_density: Union[str, Callable[[np.ndarray], np.ndarray]] = "default",
    batch_size: Optional[int] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Approximate the posterior density function for a single radiocarbon date calibration.

    This function computes an approximate posterior density for a given measured 
    radiocarbon date using a trained Bayesian Neural Network (BNN) model.  
    The approximation is done by sampling multiple stochastic realizations of 
    the BNN predictions and combining them with a likelihood term based on 
    the laboratory measurement error.

    Parameters
    ----------
    mesure : float
        The measured radiocarbon age (expressed in the F$^{14}$C domain).
    lab_error : float
        The measurement uncertainty (standard deviation) associated with the lab measurement 
        (also in the F$^{14}$C domain).
    bnn_model : object
        The trained Bayesian Neural Network model used to estimate the predictive distribution.  
        Must be compatible with `bnn_make_predictions_`.  
        This model must be an estimate of the radiocarbon calibration curve in the F$^{14}$C domain.
    nb_curves : int, optional
        Number of stochastic realizations (Monte Carlo samples) to use for 
        approximating the BNN predictive distribution.  
        Default is `100`.
    prior_density : {"default", callable}, optional
        Prior probability density over the calendar dates' domain.  
        - `"default"`: a uniform prior over `[0, 1]`, the range of the scaled calendar dates.  
        - `callable`: a custom prior density function of the form `f(dates) → np.ndarray`.  
        Default is `"default"` (the only possibility supported presently).
    batch_size : int, optional
        Batch size for model predictions, passed to the internal 
        `bnn_make_predictions_` function.  
        Default is `None`.

    Returns
    -------
    density : callable
        A function `density(dates: np.ndarray) -> np.ndarray` that computes 
        the (unnormalized) approximate posterior density of calibrated dates.  
        The dates given to this function are first scaled using `minimax_scaling` function.  
        The density is known up to a normalization constant.

    Notes
    -----
    - The posterior density is proportional to the product of the prior and the likelihood:  
      \\( p(d|m) ∝ p(d) × E_{BNN}[ \\exp(-(m - F^{14}C(d))^2 / (2σ^2)) ] \\).  
      This expectation is approximated by averaging over multiple stochastic 
      predictions of the BNN model.
    - The uniform prior currently assumes that the scaled ages' calibration domain is `[0, 1]`;  
      future implementations should allow to replace this with the actual domain limits or use
      other values to approximate these limits (e.g. training data domain bounds).  
      Another implementation improvement may be to make it possible to handle the use of a 
      `callable` which gives a custom prior density function of the form `f(dates) → np.ndarray`.  
    - The output density is **not normalized**; normalization must be handled externally 
      if necessary (e.g., via numerical integration).

    Examples
    --------
    >>> density_fn = mono_cal_date_approx_density(
    ...     mesure=0.954,
    ...     lab_error=0.002,
    ...     bnn_model=my_trained_bnn,
    ...     nb_curves=200
    ... )
    >>> ages = np.linspace(0, 1, 500)
    >>> posterior_vals = density_fn(ages)
    >>> posterior_vals.shape
    (500,)

    """
    
    # traitement de la densité à priori :
    if prior_density == "default":
        # à remplacer par min_Xtrain ou min_Xtrain_val ou min_Xtrain_val_test plus tard suivant le cas
        # ou date minimale globale possible pour la calibration
        support_lower_bound = 0.0
        # à remplacer par max_Xtrain ou max_Xtrain_val ou max_Xtrain_val_test plus tard suivant le cas
        # ou date maximale globale possible pour la calibration
        support_upper_bound = 1.0

        prior_density = lambda d: np.float64(
            (support_lower_bound <= d) * (d <= support_upper_bound)
        ) / (support_upper_bound - support_lower_bound)

    else:
        raise NotImplementedError(
            "Custom prior densities are not yet supported."
        )

    # predictions avec le modèle
    predicted = lambda d: bnn_make_predictions_(
        bnn_model=bnn_model,
        X_test=d.reshape((-1, 1)),
        iterations=nb_curves,
        batch_size=batch_size,
    )

    # densité approchée (connue à une constante près)
    density = lambda d: prior_density(d) * np.exp(
        -(mesure - predicted(d)) ** 2 / (2 * lab_error**2)
    ).mean(axis=1, dtype=np.float64) / (lab_error * np.sqrt(2 * np.pi))

    return density



# rédéfinition de la fonction mono_cal_date_approx_density pour les points milieux afin de tenir
# compte du cas où les prédictions aux points milieux sont déjà pré-calculés pour améliorer
# la rapidité des calculs en cas de plusieurs calibrations avec la même subdivision de l'intervalle [0,1]

def _mono_cal_date_approx_density_on_middle_points_(
    mesure: float,
    lab_error: float,
    bnn_model: Optional[object] = None,
    middle_points_predictions: Optional[np.ndarray] = None,
    nb_curves: int = 100,
    prior_density: Union[str, Tuple[float, float]] = "default",
    batch_size: Optional[int] = None,
    mesure_likelihood: str = [
        "gaussian_mixture",
        "curve_gaussian_approximation",
        "IntCal20",
        "exact_gaussian_density"
    ][0],  # par défaut : "gaussian_mixture"
    true_regression_function: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Approximate the posterior density for a single radiocarbon measurement using 
    precomputed predictions or a Bayesian Neural Network (BNN) model.

    This function extends `mono_cal_date_approx_density` to handle the case where
    the predictions of the calibration model at the midpoints subdivision of 
    the [0, 1] interval are **already precomputed**.
    This optimization is particularly useful when several radiocarbon dates are calibrated
    using the same subdivision of the scaled calendar-age domain `[0, 1]`, 
    avoiding repeated BNN evaluations.

    Different formulations of the measurement likelihood can be selected via 
    the `mesure_likelihood` parameter.

    Parameters
    ----------
    mesure : float
        The measured radiocarbon age expressed in the F¹⁴C domain.
    lab_error : float
        The laboratory measurement uncertainty (standard deviation), in the F¹⁴C domain.
    bnn_model : object, optional
        Trained Bayesian Neural Network model estimating the calibration curve.
        Must be compatible with `bnn_make_predictions_` and predictions expressed in the F¹⁴C domain.  
        Required if `middle_points_predictions` is not provided.
    middle_points_predictions : np.ndarray, optional
        Precomputed predictions (in the F¹⁴C domain) of the calibration model evaluated at middle points
        of the scaled date domain `[0, 1]`.  
        Expected shapes:  
        - For `"gaussian_mixture"` or `"curve_gaussian_approximation"`: `(n_points, n_curves)`  
        - For `"IntCal20"`: `(2, n_points)` where the first row is the mean curve 
          and the second row is the standard deviation.
    nb_curves : int, optional
        Number of stochastic realizations of the BNN used to approximate the predictive
        distribution when `bnn_model` is provided. Default is `100`.
    prior_density : {"default", (float, float)}, optional
        Prior probability density on the date domain.  
        - `"default"`: uniform prior over `[0, 1]`.  
        - `(min_val, max_val)`: uniform prior over a custom interval.  
        Default is `"default"`.
    batch_size : int, optional
        Batch size used for `bnn_make_predictions_` when computing BNN outputs.
    mesure_likelihood : {"gaussian_mixture", "curve_gaussian_approximation", "IntCal20", "exact_gaussian_density"}, optional
        Defines the model used for the likelihood term :  
        - `"gaussian_mixture"`: mixture of Gaussians based on BNN stochastic outputs.  
        - `"curve_gaussian_approximation"`: Gaussian approximation of the mixture 
          (Central Limit Theorem approximation).  
        - `"IntCal20"`: Gaussian approximation using IntCal20 mean and variance directly.  
        - `"exact_gaussian_density"`: exact Gaussian likelihood given a known true regression function.  
        Default is `"gaussian_mixture"`.
    true_regression_function : callable, optional
        True regression function `f(d)` required when `mesure_likelihood="exact_gaussian_density"`.
        Must return the predicted F¹⁴C values for given scaled dates.

    Returns
    -------
    density : callable
        Function `density(dates: np.ndarray) -> np.ndarray` that computes the (unnormalized) 
        posterior density of (scaled) calendar dates given the radiocarbon measurement.

    Raises
    ------
    ValueError
        If neither `bnn_model` nor `middle_points_predictions` are provided.  
        If `middle_points_predictions` is not a NumPy array.  
        If `true_regression_function` is missing when required.  
        If `mesure_likelihood` is not one of the supported options.

    Notes
    -----
    - This function is designed for **computational efficiency** when multiple 
      calibrations share the same domain discretization.
    - The prior is currently **uniform**, but this can be extended in later implementations.
    - When `"exact_gaussian_density"` is used, a deterministic regression function must be supplied.
    - The returned density is **not normalized**; normalization should be applied externally if needed.
    - The assumed calibration domain `[0, 1]` corresponds to **scaled calendar ages** obtained 
      via `minimax_scaling`.

    Mathematical Formulation
    -------------------------
    The posterior density is proportional to the product of the prior and the likelihood:
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

    3. **IntCal20 Calibration Curve**
       $$
       \\hat{L}(m|d) = \\frac{1}{\\sqrt{2\\pi(\\sigma^2 + s_{IntCal20}(d)^2)}} 
       \\exp\\left[-\\frac{(m - \\mu_{IntCal20}(d))^2}{2(\\sigma^2 + s_{IntCal20}(d)^2)}\\right]
       $$

    4. **Exact Gaussian Density**
       $$
       L(m|d) = \\hat{L}(m|d) = \\frac{1}{\\sqrt{2\\pi}\\sigma} 
       \\exp\\left[-\\frac{(m - f(d))^2}{2\\sigma^2}\\right]
       $$

    Examples
    --------
    >>> density_fn = _mono_cal_date_approx_density_on_middle_points_(
    ...     mesure=0.954,
    ...     lab_error=0.002,
    ...     bnn_model=my_bnn,
    ...     nb_curves=200,
    ...     mesure_likelihood="curve_gaussian_approximation"
    ... )
    >>> ages = np.linspace(0, 1, 1000)
    >>> posterior_vals = density_fn(ages)
    >>> posterior_vals.shape
    (1000,)
    """
    # vérification paramètres
    if bnn_model == None and type(middle_points_predictions) == 'NoneType' :
        raise ValueError(
            "At least one of 'bnn_model' or 'middle_points_predictions' must be provided (not None)."
        )
    
    if type(middle_points_predictions) != 'NoneType' and type(middle_points_predictions) != np.ndarray :
        raise ValueError(
            "'middle_points_predictions' must be of type numpy.ndarray when provided."
        )
    
    # traitement de la densité à piori :
    if prior_density == "default" :
        support_lower_bound = 0. # à remplacer par min_Xtrain ou min_Xtrain_val ou min_Xtrain_val_test plus tard suivant le cas ou date minimale gobale possible pour la calibration
        support_upper_bound = 1. # à remplacer par max_Xtrain ou max_Xtrain_val ou max_Xtrain_val_test plus tard suivant le cas ou date maximale gobale possible pour la calibration
    else :
        support_lower_bound = prior_density[0]
        support_upper_bound = prior_density[1]
        #raise NotImplementedError("La densité à piori fournie sur les dates n'est pas encore supportée")
    prior_density = lambda d: np.float64((support_lower_bound <= d) * (d <= support_upper_bound)) / (
            support_upper_bound - support_lower_bound
        )
        
    # predictions avec le modèle
    if type(middle_points_predictions) != 'NoneType' :
        predicted = lambda d : middle_points_predictions
    else : 
        predicted = lambda d : bnn_make_predictions_(
            bnn_model = bnn_model, 
            X_test = d.reshape((-1,1)), 
            iterations = nb_curves, 
            batch_size = batch_size
        )
    
    # densité approchée (connue à une constante près)
    mesure_likelihood_possible_values = [
        "gaussian_mixture", "curve_gaussian_approximation", "IntCal20", "exact_gaussian_density"
    ]
    if mesure_likelihood == mesure_likelihood_possible_values[0] :
        # on utilise le melange gaussien comme densité de la mesure
        density = lambda d : prior_density(d) * np.exp(
            -(mesure - predicted(d))**2/(2*lab_error**2)
        ).mean(axis=1, dtype=np.float64) / (lab_error * np.sqrt(2*np.pi))
        
    elif mesure_likelihood == mesure_likelihood_possible_values[1] :
        # on utilise l'approximation gaussienne du melange (TCL) comme densité de la mesure
        # revient à faire une approximation gaussienne de la courbe de calibration
        density = lambda d : prior_density(d) * np.exp(
            -(mesure - predicted(d).mean(axis=1, dtype=np.float64))**2 / (
                2 * (lab_error**2 + predicted(d).std(axis=1, dtype=np.float64)**2)
            )
        ) / np.sqrt(2 * np.pi * (lab_error**2 + predicted(d).std(axis=1, dtype=np.float64)**2))
        
    elif mesure_likelihood == mesure_likelihood_possible_values[2] :
        # similaire à l'approximation gaussienne mais avec utilisation de 
        # la courbe de calibration IntCal20 ici
        # cette option est donc à utiliser dans la calibration avec IntCal20 :
        # ici predicted(d)[0,] donne la courbe IntCal20 moyenne et predicted(d)[1,] son écart-type
        density = lambda d : prior_density(d) * np.exp(
            -(mesure - predicted(d)[0,])**2 / (
                2 * (lab_error**2 + predicted(d)[1,]**2)
            )
        ) / np.sqrt(2 * np.pi * (lab_error**2 + predicted(d)[1,]**2))
        
        
    elif mesure_likelihood == mesure_likelihood_possible_values[-1] :
        # on utilise la vraie densité gaussienne de la mesure
        # conditionnellement à la date d
        if true_regression_function == None :
            raise ValueError(
                f"'true_regression_function' must be provided when \n 'mesure_likelihood' is '{mesure_likelihood_possible_values[2]}'."
            )
        density = lambda d : prior_density(d) * np.exp(
            -(mesure - true_regression_function(d))**2 / (2*lab_error**2)
        ) / (lab_error * np.sqrt(2*np.pi))
    
    else :
        raise ValueError(
            f"'mesure_likelihood' must be one of: \n{mesure_likelihood_possible_values}"
        )
        
    return density




# version calibration simultanée de plusieurs dates

def multi_cal_date_approx_density(
    mesures: np.ndarray,
    lab_errors: np.ndarray,
    bnn_model: object,
    nb_curves: int = 100,
    prior_density: Union[str, Callable[[np.ndarray], np.ndarray]] = "default",
    batch_size: Optional[int] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Approximate the joint posterior density for multiple radiocarbon dates.

    This function generalizes the single-date calibration approach to the case 
    where several radiocarbon dates are calibrated simultaneously.  
    It computes an approximate joint posterior density over the vector of 
    (scaled) calendar dates, using a trained Bayesian Neural Network (BNN) model 
    as an estimator of the calibration curve in the F¹⁴C domain.

    Parameters
    ----------
    mesures : np.ndarray of shape (n_dates,)
        The measured radiocarbon ages expressed in the F¹⁴C domain.
    lab_errors : np.ndarray of shape (n_dates,)
        The laboratory measurement uncertainties (standard deviations), 
        also expressed in the F¹⁴C domain.
    bnn_model : object
        The trained Bayesian Neural Network model used to estimate the predictive distribution.  
        Must be compatible with `bnn_make_predictions_` and approximate the calibration curve in the F¹⁴C domain.
    nb_curves : int, optional
        Number of stochastic realizations (Monte Carlo samples) to use for 
        approximating the BNN predictive distribution.  
        Default is `100`.
    prior_density : {"default", callable}, optional
        Prior probability density over the vector of scaled calendar dates.  
        - `"default"`: a uniform prior over the hypercube `[0, 1]^n_dates`.  
        - `callable`: a custom prior density function of the form `f(dates) → np.ndarray`.  
        Default is `"default"`.
    batch_size : int, optional
        Batch size for model predictions, passed to the internal 
        `bnn_make_predictions_` function.  
        Default is `None`.

    Returns
    -------
    density : callable
        A function `density(dates: np.ndarray) -> np.ndarray` that computes 
        the (unnormalized) approximate **joint posterior density** of calibrated dates.  
        Each row in `dates` corresponds to one vector of scaled calendar dates.  
        The density is known up to a normalization constant.

    Notes
    -----
    - The posterior density is proportional to:  
      \\( p(\\mathbf{d}|\\mathbf{m}) ∝ p(\\mathbf{d}) × E_{BNN}[ \\prod_i \\exp(-(m_i - \\hat{F}^{14}C(d_i))^2 / (2σ_i^2)) ] \\).  
      The expectation over the BNN distribution is approximated by Monte Carlo averaging.
    - The `"default"` prior corresponds to a uniform independent prior over each scaled date in `[0, 1]`.
    - The output density is **not normalized**; handling normalization here via numerical integration over a 
        multi-dimensional grid is not a good idea because of the curse of dimensionality. In general, trying 
        to do so will not be necessary because the outputs densities are expected to be used within 
        a MCMC sampler (e.g. Metropolis-Hastings within Gibbs sampler).

    Raises
    ------
    NotImplementedError
        If a non-default prior density is provided (custom priors are not yet supported).

    Examples
    --------
    >>> mesures = np.array([0.954, 0.928])
    >>> lab_errors = np.array([0.002, 0.003])
    >>> density_fn = multi_cal_date_approx_density(
    ...     mesures=mesures,
    ...     lab_errors=lab_errors,
    ...     bnn_model=my_trained_bnn,
    ...     nb_curves=200
    ... )
    >>> date_grid = np.random.rand(100, 2)  # 100 candidate date vectors
    >>> posterior_vals = density_fn(date_grid)
    >>> posterior_vals.shape
    (100,)
    """
    
    dim_dates = mesures.shape[0] # = len(mesures)
    # traitement de la densité à piori :
    if prior_density == "default" :
        support_lower_bound = np.array([0.] * dim_dates) # à remplacer par min_Xtrain ou min_Xtrain_val ou min_Xtrain_val_test plus tard suivant le cas ou date minimale gobale possible pour la calibration
        support_upper_bound = np.array([1.] * dim_dates) # à remplacer par max_Xtrain ou max_Xtrain_val ou max_Xtrain_val_test plus tard suivant le cas ou date maximale gobale possible pour la calibration
        prior_density = lambda d : np.float64((support_lower_bound <= d) * (d <= support_upper_bound)).prod(axis=1)/(support_upper_bound - support_lower_bound).prod()
    else :
        raise NotImplementedError(
            "Custom prior densities for multiple dates are not yet supported."
        )
        
    # predictions avec le modèle
    # d sera une matrice (un array 2-D numpy) dont chaque ligne correspond à un vecteur de dates sur lequel sera évaluée la densité jointe
    # predicted renvoie un array 2-D de taille (d.shape[0], dim_dates, nb_curves)
    predicted = lambda d : bnn_make_predictions_(bnn_model = bnn_model, X_test = d.reshape((-1,1)), iterations = nb_curves, batch_size = batch_size).reshape((-1,dim_dates,nb_curves))
    
    # densité approchée (connue à une constante près)
    mesures_broadcasted = mesures.repeat(nb_curves).reshape((-1,nb_curves))
    lab_errors_broadcasted = lab_errors.repeat(nb_curves).reshape(-1,nb_curves)
    
    density = lambda d : prior_density(d) * np.exp(-(mesures_broadcasted - predicted(d))**2/(2*lab_errors_broadcasted**2)).prod(axis=1, dtype=np.float64).mean(axis=1, dtype=np.float64) / (lab_errors.prod() * np.sqrt(2*np.pi)**dim_dates)
    
    return density


# =============================================================================
# la densité (approchée) a posteriori des dates calibrées
# (re-définition de la fonction 'multi_cal_date_approx_density' 
# pour tenir compte du cas où on a une relation d'ordre sur les dates)
# (on tient aussi compte du cas où les prédictions aux dates 
# sont déjà disponibles dans le domaine F14C)
# (on tient également compte du cas où les prédictions aux dates 
# sont calculées par un bnn_model dans le domaine $\Delta^{14}$C : dans ce cas,
# les prédictions sont converties automatiquement dans le domaine F14C)
# =============================================================================

def _multi_cal_date_approx_density_(
    mesures: np.ndarray,
    lab_errors: np.ndarray,
    Max: Optional[float] = None,
    Min: Optional[float] = None,
    bnn_model: Optional[object] = None,
    nb_curves: int = 100,
    prior_density: str = "default",
    ordered: bool = False,
    batch_size: Optional[int] = None
) -> Callable[..., np.ndarray]:
    """
    Approximate the joint posterior density for the calibration of multiple radiocarbon
    dates, optionally incorporating an ordering constraint between the dates and 
    optionally using precomputed predictions in the F¹⁴C domain.

    This function generalizes the single–date posterior approximation to the 
    multidimensional case. It allows:  
    - calibrating several dates simultaneously,  
    - enforcing a chronological constraint (`ordered=True`),  
    - using a Bayesian Neural Network (BNN) predicting in the Δ¹⁴C domain and converting 
      predictions to the F¹⁴C domain internally,  
    - or alternatively accepting externally precomputed predictions already in the F¹⁴C domain.  

    Parameters
    ----------
    mesures : np.ndarray
        Observed radiocarbon measurements in the F¹⁴C domain. Shape `(n_dates,)`.
    lab_errors : np.ndarray
        Standard deviations of laboratory measurement errors for each date. 
        Must have shape `(n_dates,)`.
    Max : float, optional
        Maximum unscaled calendar age used during training of the BNN calibration curve.  
        Required if `bnn_model` is provided (used for reversing the scaling of input dates).
    Min : float, optional
        Minimum unscaled calendar age used during training of the BNN calibration curve.  
        Required if `bnn_model` is provided (used for reversing the scaling of input dates).
    bnn_model : object, optional
        Trained Bayesian Neural Network model predicting Δ¹⁴C.  
        If not provided, the returned density function expects precomputed predictions 
        in the F¹⁴C domain.
    nb_curves : int, optional
        Number of Monte Carlo draws from the BNN used to approximate predictive 
        distribution. Default is `100`.
    prior_density : {"default"}, optional
        Prior over the scaled dates.  
        `"default"`: independent uniform prior over `[0,1]` for each date.  
        No other option is currently supported.
    ordered : bool, optional
        If `True`, enforces a strict chronological ordering constraint on the dates:
        d₀ < d₁ < … < d_{n-1}.  
        Implemented by multiplying the prior by an indicator function.
        Default is `False`.
    batch_size : int, optional
        Batch size for prediction calls to `bnn_make_predictions_`. Default `None`.

    Returns
    -------
    density : callable
        - If `bnn_model` is provided:  
          `density(dates: np.ndarray) -> np.ndarray`
        - If `bnn_model` is None:  
          `density(dates: np.ndarray, predicted_d: np.ndarray) -> np.ndarray`

        The returned function evaluates an **unnormalized** approximation of the 
        joint posterior density of the calendar dates.

    Notes
    -----
    - The posterior density is proportional to:  
      \\( p(\\mathbf{d}|\\mathbf{m}) ∝ p(\\mathbf{d}) × E_{BNN}[ \\prod_i \\exp(-(m_i - \\hat{F}^{14}C(d_i))^2 / (2σ_i^2)) ] \\).  
      The expectation over the BNN distribution is approximated by Monte Carlo averaging.
    - When `bnn_model` is provided, it predicts Δ¹⁴C values which are internally 
      converted to F¹⁴C using `d14c_to_f14c`.
    - The transformation from scaled dates d ∈ [0,1] to calendar ages uses the 
      inverse minimax scaling defined by the `Min` and `Max` arguments.
    - The `"default"` prior assumes dates are independently uniform over `[0,1]` 
      (optionally conditioned on respecting chronological order).
    - The returned density is **not normalized**.  
      Normalization in multiple dimensions is not recommended due to the 
      **curse of dimensionality**, and this density is intended to be used 
      within an **MCMC sampler** (e.g., Metropolis–Hastings within Gibbs).
    - For `bnn_model=None`, the user must provide F¹⁴C predictions of shape  
      `(n_eval_points, n_dates, nb_curves)`.

    """
    
    dim_dates = mesures.shape[0] # = len(mesures)
    # traitement de la densité à piori :
    if prior_density == "default" :
        support_lower_bound = np.array([0.] * dim_dates) # à remplacer par min_Xtrain ou min_Xtrain_val ou min_Xtrain_val_test plus tard suivant le cas ou date minimale gobale possible pour la calibration
        support_upper_bound = np.array([1.] * dim_dates) # à remplacer par max_Xtrain ou max_Xtrain_val ou max_Xtrain_val_test plus tard suivant le cas ou date maximale gobale possible pour la calibration
        if not ordered : 
            prior_density = lambda d : np.float64((support_lower_bound <= d) * (d <= support_upper_bound)).prod(axis=1)/(support_upper_bound - support_lower_bound).prod()
        else : 
            prior_density = lambda d : np.float64(np.argsort(d) == np.arange(dim_dates)).prod(axis=1) * np.float64((support_lower_bound <= d) * (d <= support_upper_bound)).prod(axis=1)/(support_upper_bound - support_lower_bound).prod()
    else :
        raise NotImplementedError(
            "Custom prior densities are not supported yet."
        )
        
    # densité approchée (connue à une constante près)
    mesures_broadcasted = mesures.repeat(nb_curves).reshape((-1,nb_curves))
    lab_errors_broadcasted = lab_errors.repeat(nb_curves).reshape(-1,nb_curves)
    
    if bnn_model != None :
        if Max == None or Min == None :
            raise ValueError(
                "Arguments 'Max' and 'Min' must be provided when 'bnn_model' is used."
            )
        # predictions avec le modèle
        # d sera une matrice (un array 2-D numpy) dont chaque ligne correspond à un vecteur de dates sur lequel est évaluée la densité jointe
        # predicted renvoie un array 2-D de taille (d.shape[0], dim_dates, nb_curves)
        predicted = lambda d : bnn_make_predictions_(bnn_model = bnn_model, X_test = d.reshape((-1,1)), iterations = nb_curves, batch_size = batch_size).reshape((-1,dim_dates,nb_curves))
    
        density = lambda d : (
            prior_density(d) * 
            np.exp(-(
                mesures_broadcasted - 
                d14c_to_f14c(
                    d14c = predicted(d),
                    teta = minimax_scaling_reciproque(
                        x = d.reshape((-1,1)),
                        Max = Max,
                        Min = Min
                    ).repeat(nb_curves).reshape((-1,dim_dates,nb_curves))
                )
            )**2/(2*lab_errors_broadcasted**2)).prod(axis=1, dtype=np.float64).mean(axis=1, dtype=np.float64) / (lab_errors.prod() * np.sqrt(2*np.pi)**dim_dates)
        )
    
    else :
        # les predictions associées à d seront fournies en entrées 
        # et seront un ndarray de shape (d.shape[0], d.shape[1], nb_curves) # d.shape[1] = dim_dates
        density = lambda d, predicted_d : (
            prior_density(d) * 
            np.exp(-(
                mesures_broadcasted - predicted_d
            )**2/(2*lab_errors_broadcasted**2)).prod(axis=1, dtype=np.float64).mean(axis=1, dtype=np.float64) / (lab_errors.prod() * np.sqrt(2*np.pi)**dim_dates)
        )
     
    return density







# =============================================================================
# simulation suivant la densité (approchée) a posteriori des dates calibrées
# =============================================================================

def mono_cal_date_approx_density_sample(
    density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    nb_intervals: int = 1000,
    support_bounds: Tuple[float, float] = (0.0, 1.0),
    subdivision_components: Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = None,
    sample_size: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draw samples from an univariate, unnormalized posterior density using a 
    piecewise-constant approximation over a regular grid.

    This function is typically used in Bayesian radiocarbon calibration to draw
    samples from the approximate posterior distribution of a **single** calibrated
    date, when the posterior density is only available up to a multiplicative
    constant or when *direct* numerical integration is not desirable.

    Parameters
    ----------
    density : callable or None
        A function evaluating the unnormalized posterior density on an array
        of points. Required unless `subdivision_components` is supplied.

    nb_intervals : int
        Number of subintervals defining the grid approximation of the density.

    support_bounds : tuple of float
        Lower and upper bounds of the support. Only (0, 1) is currently supported.

    subdivision_components : tuple of numpy.ndarray or None
        Optional tuple (interval_bounds, midpoints, midpoint_densities).
        If provided, these arrays are reused directly.

    sample_size : int
        Number of posterior samples to generate.

    Returns
    -------
    tuple
        A tuple (d, unnorm_prob, norm_prob) with:  
            - d : `ndarray of shape (sample_size,)`, 
                the generated samples.  
            - unnorm_prob : `ndarray of shape (sample_size,)`, 
                the unnormalized density values at the selected midpoints.  
            - norm_prob : `ndarray of shape (sample_size,)`, 
                the normalized discrete probabilities associated with the chosen intervals 
                (see the notes below).  

    Notes
    -----
    **1. Posterior density availability**
        
        The function assumes that the posterior density `p(d | m)` is only known
        through an unnormalized function:

            f(d) ∝ p(d | m)

        This is the case when the density results from Monte Carlo averaging over a
        Bayesian Neural Network (BNN), where:

            f(d) = p(d) × E_BNN[ exp(-(m - F¹⁴C(d))² / (2σ²)) ] / (σ √(2π))

        Since the density is unnormalized, classical continuous inversion sampling 
        is not possible. Instead, a piecewise-constant discretization is used.

    **2. Numerical approximation**
        
        The interval [0, 1] is subdivided into `nb_intervals` equal subintervals.
        On each subinterval, the density is approximated by its midpoint value:

            f(d) ≈ f(d_j*)   for d in interval j

        yielding discrete weights:

            p_j = f(d_j*) / Σ_k f(d_k*)

        which define a categorical distribution over the intervals.

    **3. Sampling algorithm**
        
        Sampling is performed as follows:

        1. Compute (or reuse) midpoints `d_j*` and their unnormalized densities.
        2. Normalize these densities to obtain probabilities over subintervals.
        3. Draw an interval index J according to these probabilities.
        4. Draw a uniform sample on the chosen interval:

               d ~ Uniform(interval_bounds[J-1], interval_bounds[J])

        This yields samples approximately distributed according to the target
        posterior density.

    **4. Precomputed subdivision**  

        If `subdivision_components = (bounds, midpoints, densities)` is supplied,
        the function skips all density evaluations and directly reuses the
        piecewise-constant representation.  
        This is useful when repeatedly sampling from the same density is needed, 
        e.g. inside an MCMC procedure.

    **5. Support and limitations**

        - Only the default support (0, 1) is currently implemented.
        - The method is *univariate*.  
          Multivariate posterior sampling must instead rely on MCMC (e.g. 
          Metropolis–Hastings within Gibbs), since grid-based density approximation 
          in higher dimension is impractical due to the curse of dimensionality.
        - The density is never normalized by continuous integration (on purpose),
          this yields a converging approximation of f(d) when the number of subintervals,
          `nb_intervals`, tends to infinity.
    
    Examples
    --------
    >>> density = lambda x: np.exp(-(x-0.4)**2 / 0.01)
    >>> d, f_unnorm, f_norm = mono_cal_date_approx_density_sample(
    ...     density=density, 
    ...     sample_size=5
    ... )
    """

    # traitement du support et contrôle des arguments fournis
    if support_bounds != (0,1) :
        raise NotImplementedError(
            "Only support (0,1) is currently supported."
        )
        
    if density == None and subdivision_components == None :
        raise ValueError(
            "At least one of 'density' or 'subdivision_components' must be provided."
        )
        
    if subdivision_components != None and len(subdivision_components) != 3 :
        raise ValueError(
            "'subdivision_components' must be a tuple of three arrays: bounds, midpoints, densities."
        )
        
    if subdivision_components != None : 
        
        intervals_bounds = subdivision_components[0]
        middle_points = subdivision_components[1]
        middle_points_density = subdivision_components[2]
        nb_intervals = len(middle_points)
        
    else :
    
        # subdivision du support en nb_intervals : calcul des bornes des sous-intervalles
        intervals_bounds = np.linspace(support_bounds[0], support_bounds[1], nb_intervals+1, dtype=np.float64)
        
        # évaluation de la densité aux points milieu
        # midle_points = (support_bounds[1] - support_bounds[0])/(2*nb_intervals) + intervals_bounds[:-1]
        middle_points = (intervals_bounds[:-1] + intervals_bounds[1:])/2
        middle_points_density = density(middle_points)
        
    # pour tirer une date d suivant la densité voulue, on procède comme suit :
    #     *) tirer un indice j dans dans {1, 2, ..., nb_intervals} muni de la probabilité 
    #         middle_points_density/middle_points_density.sum() (on peut se contenter de la 
    #         probabilité non normalisée middle_points_density)
    #     *) tirer d suivant la loi uniforme sur le j ième intervalle [intervals_bounds[j-1], intervals_bounds[j]]
    rng = np.random.default_rng()
    probabilities = middle_points_density/middle_points_density.sum()
    # while probabilities.sum() != 1 :
    #     probabilities = probabilities/probabilities.sum()
    j = 1 + rng.choice(nb_intervals, size = sample_size, p = probabilities) # le +1 permet d'avoir j entre 1 et nb_intervals au lieu de 0 et nb_intervals - 1
    u = rng.random(size = sample_size) # loi uniforme sur [0,1]
    d = (intervals_bounds[j] - intervals_bounds[j-1]) * u + intervals_bounds[j-1] # équivalent aussi (support_bounds[1] - support_bounds[0])/nb_intervals * u + intervals_bounds[j-1]
    
    # on retourne d et sa probabilité (non normalisée et normalisée)
    return d, middle_points_density[j-1], probabilities[j-1]

# =============================================================================
# approximation de la fonction de répartition a posteriori des dates calibrées
# =============================================================================

def mono_cal_date_approx_cumulative_fct(
    density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    nb_intervals: int = 1000,
    support_bounds: Tuple[float, float] = (0, 1),
    subdivision_components: Optional[Union[Tuple[np.ndarray, np.ndarray, np.ndarray], List[np.ndarray]]] = None
) -> Callable[[float], float]:
    """
    Approximate the posterior cumulative distribution function (CDF) for a single calibrated radiocarbon date.

    This function builds a continuous CDF from either:  
      - a callable density function `density`, or  
      - precomputed subdivision components (interval bounds, middle points, densities at middle points).

    The continuity of the resulting CDF is a direct mathematical consequence of the
    sampling strategy used for generating posterior samples: each interval is chosen
    with probability proportional to its middle-point density, and points within the
    interval are drawn uniformly. This results in a CDF that increases linearly
    within each interval.

    Parameters
    ----------
    density : callable, optional
        A function `density(dates: np.ndarray) -> np.ndarray` computing the 
        approximate posterior density at given points. Required if `subdivision_components` is None.
    nb_intervals : int, default=1000
        Number of subintervals to discretize the support for the density approximation.
        Overrided internally if `subdivision_components` is provided.
    support_bounds : tuple of float, optional
        Tuple `(min, max)` specifying the bounds of the scaled date support. Only `(0,1)` is
        currently supported. Default is `(0,1)`.
    subdivision_components : tuple or list of three numpy.ndarray, optional
        Precomputed components for the density approximation:  
            - array of interval bounds  
            - array of middle points  
            - array of density values at middle points  
        If provided, `density` is ignored and `nb_intervals` is overrided.

    Returns
    -------
    cumulative_density : callable
        A function `CDF(date: float) -> float` returning the approximate
        posterior cumulative probability for the given date.

    Raises
    ------
    NotImplementedError
        If `support_bounds` is not `(0, 1)`.
    ValueError
        If neither `density` nor `subdivision_components` are provided, or if
        `subdivision_components` does not contain exactly three arrays.

    Notes
    -----
    - Let the support $[a,b]$ be subdivided into N intervals $[x_j, x_{j+1}]$ with
      middle points $m_j = \\frac{x_j + x_{j+1}}{2}$, and let $f_j = density(m_j)$.
      The approximate $CDF$ at a point $d \in [x_j, x_{j+1}]$ is:
        $$
        CDF(d) = \sum_{i=1}^{j-1} \\dfrac{f_i}{\sum_{k=1}^N f_k}  + 
        \\dfrac{f_j}{\sum_{k=1}^N f_k} \\dfrac{d - x_j}{x_{j+1} - x_j}
        $$  
      where the first term sums contributions from previous intervals, and the second
      term accounts for the uniform distribution inside the current interval.  
    - Continuity of the CDF arises naturally from the uniform distribution inside
      intervals.  
    - This approach provides a simple and fast approximation suitable for sampling
      posterior dates. For N → ∞, the discrete sum converges to the integral
      of the continuous piecewise density function.

    Examples
    --------
    >>> density_fn = mono_cal_date_approx_density(
    ...     mesure=0.954,
    ...     lab_error=0.002,
    ...     bnn_model=my_trained_bnn,
    ...     nb_curves=200
    ... )
    >>> cdf_fn = mono_cal_date_approx_cumulative_fct(density=density_fn, nb_intervals=1000)
    >>> scaled_date = 0.25
    >>> cdf_value = cdf_fn(scaled_date)
    >>> isinstance(cdf_value, float)
    True

    """

    # traitement du support et contrôle des arguments fournis
    if support_bounds != (0,1) :
        raise NotImplementedError(
            "Only the default support (0,1) is currently supported"
        )
        
    if density == None and subdivision_components == None :
        raise ValueError(
            "At least one of 'density' or 'subdivision_components' must be provided (not None)"
        )
        
    if subdivision_components != None and len(subdivision_components) != 3 :
        raise ValueError(
            "'subdivision_components' must be a tuple or list of length 3 containing arrays: interval bounds, middle points, and densities at middle points"
        )
        
    if subdivision_components != None : 
        
        intervals_bounds = subdivision_components[0]
        middle_points = subdivision_components[1]
        middle_points_density = subdivision_components[2]
        nb_intervals = len(middle_points)
        
    else :
    
        # subdivision du support en nb_intervals : calcul des bornes des sous-intervalles
        intervals_bounds = np.linspace(support_bounds[0], support_bounds[1], nb_intervals+1, dtype=np.float64)
        
        # évaluation de la densité aux points milieu
        # midle_points = (support_bounds[1] - support_bounds[0])/(2*nb_intervals) + intervals_bounds[:-1]
        middle_points = (intervals_bounds[:-1] + intervals_bounds[1:])/2
        middle_points_density = density(middle_points)
    
    # la borne inférieure de l'intervalle qui contient la date d sur laquelle évaluer la fonction
    lower_test = lambda d : intervals_bounds[:-1] <= d 
    idx = lambda d : np.where(lower_test(d))[0][-1] #-1 car c'est la dernière borne pour laquelle le test ci-dessus vaut true
    # NB : np.where renvoie un tuple d'array (ici c'est un tuple avec un seul array) et c'est le premier élément du tuple qui nous intéresse ici
    
    # calcul de la densité cumulée au point d 
    h = (support_bounds[1] - support_bounds[0])/nb_intervals
    cumulative_density = lambda d : (middle_points_density[idx(d)]*(d - intervals_bounds[idx(d)])/h + middle_points_density[:idx(d)].sum())/middle_points_density.sum()
    
    return cumulative_density


def mono_cal_date_approx_vect_cumulative_fct(density=None, nb_intervals=1000, support_bounds=(0,1), subdivision_components=None):
    
    # traitement du support et contrôle des arguments fournis
    if support_bounds != (0,1) :
        raise NotImplementedError("Le support fourni n'est pas encore supporté")
        
    if density == None and subdivision_components == None :
        raise ValueError("au moins l'un des arguments 'density' ou 'subdivision_components' doit être fourni (!= None)")
        
    if subdivision_components != None and len(subdivision_components) != 3 :
        raise ValueError("'subdivision_components' doit être un tuple ou une liste de taille 3 contenant 3 arrays : celui des bornes de sous intervalles, de points milieux et de densités aux points milieux")
        
    if subdivision_components != None : 
        
        intervals_bounds = subdivision_components[0]
        middle_points = subdivision_components[1]
        middle_points_density = subdivision_components[2]
        nb_intervals = len(middle_points)
        
    else :
    
        # subdivision du support en nb_intervals : calcul des bornes des sous-intervalles
        intervals_bounds = np.linspace(support_bounds[0], support_bounds[1], nb_intervals+1, dtype=np.float64)
        
        # évaluation de la densité aux points milieu
        # midle_points = (support_bounds[1] - support_bounds[0])/(2*nb_intervals) + intervals_bounds[:-1]
        middle_points = (intervals_bounds[:-1] + intervals_bounds[1:])/2
        middle_points_density = density(middle_points)
    
    # la borne inférieure de l'intervalle qui contient la date d sur laquelle évaluer la fonction
    lower_test = lambda d : intervals_bounds[:-1] <= d.reshape((-1,1)) 
    
    def idx(d) : 
        condition_res = np.where(lower_test(d))
        idx_of_idx = []
        for i in range(len(d)) :
            idx_of_idx.append(np.where(condition_res[0]==i)[0][-1]) #-1 car c'est la dernière borne pour laquelle le test ci-dessus vaut true
            # NB : np.where renvoie un tuple d'array (ici c'est un tuple avec un seul array) et c'est le premier élément du tuple qui nous intéresse ici
        idx_of_idx = np.array(idx_of_idx)
        
        res_idx = condition_res[1][idx_of_idx]
        return res_idx
    
    # # autre possibilité de calcul de la fonction idx sans recourir à une boucle :
    
    # # test sur les bornes inf et sup des intervalles qui contiennent les dates
    # lower_test = lambda d : intervals_bounds[:-1] <= d.reshape((-1,1))
    # intervals_bounds[-1] = intervals_bounds[-1] + 1 # on ajoute un nb positif à la borne sup du support pour donner un sens a inf <= d < sup pour tout d du support (sinon d = sup poserait problème)
    # upper_test = lambda d : d.reshape((-1,1)) < intervals_bounds[1:]
    
    # # enfin on trouverait les indices des bornes inf des intervalles comme suit :
    # idx = lambda d : np.where(lower_test(d)*upper_test(d))[1]
    # # NB : np.where renvoie un tuple d'array (ici c'est un tuple avec un 2 arrays : array des dimensions et array des indices) et c'est le deuxième élément du tuple qui nous intéresse ici
    
    # calcul de la densité cumulée au point d 
    h = (support_bounds[1] - support_bounds[0])/nb_intervals
    
    def cumulative_density(d):
        res_idx = idx(d)
        inf_cumulative_density = []
        for idx_d in res_idx :
            inf_cumulative_density.append(middle_points_density[:idx_d].sum())
        inf_cumulative_density = np.array(inf_cumulative_density)

        vect_cumulative_density = (middle_points_density[res_idx]*(d - intervals_bounds[res_idx])/h + inf_cumulative_density)/middle_points_density.sum()
        return vect_cumulative_density
    
    return cumulative_density


# =============================================================================
# approximation de la fonction quantile a posteriori des dates calibrées
# =============================================================================

def mono_cal_date_discrete_approx_quantile_fct(density=None, nb_intervals=1000, support_bounds=(0,1), subdivision_components=None):
    
    # traitement du support et contrôle des arguments fournis
    if support_bounds != (0,1) :
        raise NotImplementedError("Le support fourni n'est pas encore supporté")
        
    if density == None and subdivision_components == None :
        raise ValueError("au moins l'un des arguments 'density' ou 'subdivision_components' doit être fourni (!= None)")
        
    if subdivision_components != None and len(subdivision_components) != 3 :
        raise ValueError("'subdivision_components' doit être un tuple ou une liste de taille 3 contenant 3 arrays : celui des bornes de sous intervalles, de points milieux et de densités aux points milieux")
        
    if subdivision_components != None : 
        
        intervals_bounds = subdivision_components[0]
        middle_points = subdivision_components[1]
        middle_points_density = subdivision_components[2]
        # nb_intervals = len(middle_points)
        
    else :
    
        # subdivision du support en nb_intervals : calcul des bornes des sous-intervalles
        intervals_bounds = np.linspace(support_bounds[0], support_bounds[1], nb_intervals+1, dtype=np.float64)
        
        # évaluation de la densité aux points milieu
        # midle_points = (support_bounds[1] - support_bounds[0])/(2*nb_intervals) + intervals_bounds[:-1]
        middle_points = (intervals_bounds[:-1] + intervals_bounds[1:])/2
        middle_points_density = density(middle_points)
    
    # calcul de la fonction de répartition aux points milieu
    middle_points_density_cumsum = middle_points_density.cumsum()
    middle_points_cumulative_density = (middle_points_density/2 + np.concatenate((np.array([0]), middle_points_density_cumsum[:-1]), dtype=np.float64)) / middle_points_density.sum()
    
    # Enfin on peut déterminer les fonctions de répartition et quantile ainsi discrétisées
    discrete_cumulative_density = np.concatenate((np.array([0]), middle_points_cumulative_density, np.array([1])), dtype=np.float64)
    discrete_quantiles = np.concatenate((np.array([support_bounds[0]]), middle_points, np.array([support_bounds[1]])), dtype=np.float64)
    
    # Pour terminer, on définit la fonction quantile qui retournera le quantile d'ordre alpha sur base de cette discrétisation
    idx = lambda alpha : np.where(discrete_cumulative_density >= alpha)[0][0] # les densités cumulées et les quantiles étant ordonnés, il suffit de prendre le premier indice (autrement il aurait fallu ordonner d'abord)
    discrete_alpha_quantile = lambda alpha : discrete_quantiles[idx(alpha)]
    
    return discrete_alpha_quantile
    
def mono_cal_date_exact_approx_quantile_fct(density=None, nb_intervals=1000, support_bounds=(0,1), subdivision_components=None):
    
    # traitement du support et contrôle des arguments fournis
    if support_bounds != (0,1) :
        raise NotImplementedError("Le support fourni n'est pas encore supporté")
        
    if density == None and subdivision_components == None :
        raise ValueError("au moins l'un des arguments 'density' ou 'subdivision_components' doit être fourni (!= None)")
        
    if subdivision_components != None and len(subdivision_components) != 3 :
        raise ValueError("'subdivision_components' doit être un tuple ou une liste de taille 3 contenant 3 arrays : celui des bornes de sous intervalles, de points milieux et de densités aux points milieux")
        
    if subdivision_components != None : 
        
        intervals_bounds = subdivision_components[0]
        middle_points = subdivision_components[1]
        middle_points_density = subdivision_components[2]
        nb_intervals = len(middle_points)
        
    else :
    
        # subdivision du support en nb_intervals : calcul des bornes des sous-intervalles
        intervals_bounds = np.linspace(support_bounds[0], support_bounds[1], nb_intervals+1, dtype=np.float64)
        
        # évaluation de la densité aux points milieu
        # midle_points = (support_bounds[1] - support_bounds[0])/(2*nb_intervals) + intervals_bounds[:-1]
        middle_points = (intervals_bounds[:-1] + intervals_bounds[1:])/2
        middle_points_density = density(middle_points)
    
    # calcul de la fonction de répartition aux "nb_intervals + 1" bornes des sous-intervalles du support
    middle_points_density_sum = middle_points_density.sum()
    middle_points_density_cumsum = middle_points_density.cumsum()
    intervals_bounds_cumulative_density = np.concatenate((np.array([0]), middle_points_density_cumsum / middle_points_density_sum), dtype=np.float64)
    
    # on met à 1 les éventuelles valeurs de la densité cumulée (fonction de répartition) qui seraient supérieures à 1 à cause des erreurs d'arrondis
    intervals_bounds_cumulative_density = np.where(intervals_bounds_cumulative_density > 1., 1., intervals_bounds_cumulative_density)
    
    # on rajoute 0 comme premier élément de 'middle_points_density_cumsum' pour que cela puisse bien marcher avec les indices lors du calcul de 'd_alpha' plus loin
    # en effet on voudra que si idx_borne_inf = 0 (donc la borne inf de l'intervalle = borne inf du support), alors middle_points_density_cumsum[idx_borne_inf] doit donner 0
    middle_points_density_cumsum = np.concatenate((np.array([0]), middle_points_density_cumsum), dtype=np.float64)
    # (attention : len(middle_points_density_cumsum) est nb_intervals + 1 désormais alors que len(middle_points_density) vaut toujours nb_intervals, ce qui fait en sorte que 
    # middle_points_density[idx_borne_inf] donne bien la densité (non normalisée) au point milieu de l'intervalle de borne inf = intervals_bounds[idx_borne_inf], ce qui est 
    # exactement le résultat voulus)
    
    # on peut enfin calculer notre fonction quantile
    def exact_alpha_quantile(alpha) :
        try :
            # Si l'une des "nb_intervals + 1" bornes a une densité cumulée (fonction de répartition) qui vaut alpha, alors c'est le quantile recherché
            idx_borne = np.where(intervals_bounds_cumulative_density == alpha)[0][0]
            return intervals_bounds[idx_borne]
        except IndexError :
            # Sinon, aucune des bornes n'est le quantile :
            # on cherche alors la borne inf de l'intervalle qui contient notre quantile : c'est la dernère borne avec une densité cumulée < alpha
            idx_borne_inf = np.where(intervals_bounds_cumulative_density < alpha)[0][-1]
            
            # on calcule alors le quantile d'ordre alpha, d_alpha, suivant la formule de la fonction quantile 
            h = (support_bounds[1] - support_bounds[0])/(nb_intervals)
            d_alpha = intervals_bounds[idx_borne_inf] + h / middle_points_density[idx_borne_inf] * (alpha * middle_points_density_sum - middle_points_density_cumsum[idx_borne_inf])
            return d_alpha
    
    # Enfin on peut retourner le fonction quantile ainsi construite
    return exact_alpha_quantile

# =============================================================================
# intervalles de crédibilité et régions HPD 
# (cas de la calibration individuelle)
# =============================================================================

def optimise_credible_interval(quantile, alpha) :
    interval_length = lambda beta : quantile(1 - alpha + beta) - quantile(beta)
    beta_opt = minimize(fun = interval_length, x0 = np.array([alpha/2]), method = 'Nelder-Mead', bounds = [(0.,alpha)])
    return beta_opt

def compute_HPD_regions(alpha, density=None, nb_intervals=1000, support_bounds=(0,1), subdivision_components=None):
    
    # traitement du support et contrôle des arguments fournis
    if support_bounds != (0,1) :
        raise NotImplementedError("Le support fourni n'est pas encore supporté")
        
    if density == None and subdivision_components == None :
        raise ValueError("au moins l'un des arguments 'density' ou 'subdivision_components' doit être fourni (!= None)")
        
    if subdivision_components != None and len(subdivision_components) != 3 :
        raise ValueError("'subdivision_components' doit être un tuple ou une liste de taille 3 contenant 3 arrays : celui des bornes de sous intervalles, de points milieux et de densités aux points milieux")
        
    if subdivision_components != None : 
        
        intervals_bounds = subdivision_components[0]
        middle_points = subdivision_components[1]
        middle_points_density = subdivision_components[2]
        # nb_intervals = len(middle_points)
        
    else :
    
        # subdivision du support en nb_intervals : calcul des bornes des sous-intervalles
        intervals_bounds = np.linspace(support_bounds[0], support_bounds[1], nb_intervals+1, dtype=np.float64)
        
        # évaluation de la densité aux points milieu
        # midle_points = (support_bounds[1] - support_bounds[0])/(2*nb_intervals) + intervals_bounds[:-1]
        middle_points = (intervals_bounds[:-1] + intervals_bounds[1:])/2
        middle_points_density = density(middle_points)
    
    # on reordonne les intervalles (les densités) suivant la valeur de leurs densités, dans l'ordre décroissante
    sorted_desc_index = np.argsort(middle_points_density)[::-1]
    sorted_desc_middle_points_density = middle_points_density[sorted_desc_index]
    
    # on calcule la densité cumulée des densités ainsi reordonnéees et on les renormalise pour 
    # pouvoir avoir 1 comme dernier élément comme dans unvecteur de fonction de répartition
    scaling_weight = sorted_desc_middle_points_density.sum()
    sorted_desc_middle_points_density_cumsum_scaled = sorted_desc_middle_points_density.cumsum()/scaling_weight
    
    # au cas où il y a des valeurs > 1 dans sorted_desc_middle_points_density_cumsum_scaled (à cause des erreurs d'arrondis), on les met à 1
    sorted_desc_middle_points_density_cumsum_scaled = np.where(sorted_desc_middle_points_density_cumsum_scaled > 1., 1., sorted_desc_middle_points_density_cumsum_scaled)
    
    # on calcule le mode a posteriori
    # (TODO : voir plus tard si nécessaire de calculer plusieurs modes de même densité HPD éventuellement)
    calage_posterior_mode_density = middle_points_density[sorted_desc_index[0]]/scaling_weight
    calage_posterior_mode = middle_points[sorted_desc_index[0]]
    
    
    # on peut alors regarder à partir de quand on dépasse le seuil de 1 - alpha
    where_idx = np.where(sorted_desc_middle_points_density_cumsum_scaled >= 1 - alpha)[0][0]
    k_1_alpha = sorted_desc_middle_points_density[where_idx]/scaling_weight # = quantile d'ordre alpha du vecteur des densités
    
    # on recupère alors les (indices des) intervalles formant la région HPD sélectionnés parmi les nb_intervals : 
    # les indices sont reordonnés du plus petit au plus grand
    selected_intervals_idx = np.sort(sorted_desc_index[:(where_idx+1)])
    
    # maintenant, il suffit de déterminer les intervalles HPD sélectionnés, tout en regroupant les parties connexes 
    # et en calculant la densité de chaque partie connexe ainsi constituée
    connexe_intervals = []
    connexe_intervals_density = []
    l = len(selected_intervals_idx)
    i = 0
    while i < l :
        first = selected_intervals_idx[i]
        last = first
        j = i+1
        while j < l :
            current = selected_intervals_idx[j]
            if current - last == 1 :
                last = current
                j = j+1
            else :
                break
        if first == last :
            connexe_intervals.append([intervals_bounds[first], intervals_bounds[first+1]])
            connexe_intervals_density.append(middle_points_density[first]/scaling_weight)
        else :
            connexe_intervals.append([intervals_bounds[first], intervals_bounds[last+1]])
            connexe_intervals_density.append(middle_points_density[first:(last+1)].sum()/scaling_weight)
        i = j
        
    # on peut alors retourner la région HPD sous formes des parties connexes ainsi que les densités
    # de différentes parties connexes
    # en bonus, on ajoute le seuil k_1_alpha permettant de déterminer la région HPD
    # on ajoute aussi le mode calage_mode le plus plaussible : la date calibrée la plus probable a posteriori
    return {
            "calage_posterior_mode" : calage_posterior_mode,
            "calage_posterior_mode_density" : calage_posterior_mode_density,
            "connexe_HPD_intervals" : connexe_intervals, 
            "connexe_HPD_intervals_density" : connexe_intervals_density, 
            "HPD_threshold" : k_1_alpha
        }



__all__ = [
    "mono_cal_date_approx_density",
    "_mono_cal_date_approx_density_on_middle_points_",
    "multi_cal_date_approx_density",
    "_multi_cal_date_approx_density_",
    "mono_cal_date_approx_density_sample",
    "mono_cal_date_approx_cumulative_fct",
    "mono_cal_date_approx_vect_cumulative_fct",
    "mono_cal_date_discrete_approx_quantile_fct",
    "mono_cal_date_exact_approx_quantile_fct",
    "optimise_credible_interval",
    "compute_HPD_regions"

]