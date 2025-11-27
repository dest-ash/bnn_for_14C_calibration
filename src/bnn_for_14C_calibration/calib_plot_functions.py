# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import minimize

from matplotlib import pyplot as plt
import pandas as pd


from .bnn_models_built_in_utils import (
    #bnn_make_predictions_, 
    bnn_load_predictions_
)

from .utils import (
    get_lib_data_paths,
    minimax_scaling_reciproque, minimax_scaling,
    f14c_to_c14, #f14csig_to_c14sig,
    c14_to_f14c, c14sig_to_f14csig,
    f14c_to_d14c, f14csig_to_d14csig,
    d14c_to_f14c, d14csig_to_f14csig,
    d14c_to_c14, d14csig_to_c14sig
)

from .calibration import (
    concatenate_curves_parts
)


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
# Représentations graphiques 
# =============================================================================



# =============================================================================
# Courbes de calibration
# =============================================================================


# courbe IntCal20
def add_IntCal20_curve(
    ax: plt.Axes = None,
    figsize: tuple = None,
    color: str = "green",
    alpha: float = 0.4,
    incertitude: bool = True,
    sigma_length: int = 1,
    Min_x: float = None,
    Max_x: float = None,
    Min_y: float = None,
    Max_y: float = None,
    invert_xaxis: bool = True,
    domaine: str = ['delta14c', 'c14', 'f14c'][0]
) -> plt.Axes:
    """
    Add the IntCal20 radiocarbon calibration curve to a Matplotlib axis.

    The IntCal20 curve is plotted in one of three possible domains:

    - ``'c14'``: radiocarbon ages (BP)
    - ``'f14c'``: Fraction Modern F¹⁴C
    - ``'delta14c'``: Δ¹⁴C (per mil)

    Optional uncertainty bands corresponding to ``sigma_length`` standard
    deviations may be added around the curve.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, optional
        Axis on which to draw the curve. If None, uses ``plt.gca()``.
    figsize : tuple or None, optional
        Figure size passed to pandas' ``DataFrame.plot`` method. Only used
        when ``ax`` is None.
    color : str, optional
        Main curve color (default: ``"green"``).
    alpha : float, optional
        Transparency of the uncertainty band (default: 0.4).
    incertitude : bool, optional
        If True, plot the ±σ uncertainty region around the curve.
    sigma_length : int, optional
        Number of standard deviations for the uncertainty band.
    Min_x, Max_x : float or None, optional
        Optional x-axis limits; when both provided the axis limits are set.
    Min_y, Max_y : float or None, optional
        Optional y-axis limits; when both provided the axis limits are set.
    invert_xaxis : bool, optional
        If True (default), invert the x-axis so that time decreases to the right,
        consistent with radiocarbon convention.
    domaine : {'delta14c', 'c14', 'f14c'}
        Domain in which to plot the curve. Default is ``'delta14c'``.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plotted calibration curve.

    """

    
    # courbe IntCal20
    IntCal20_file_path = IntCal20_dir / "IntCal20_completed.csv"
    IntCal20 = pd.read_csv(IntCal20_file_path, sep =",")
    
    # Création de l'axe si pas fourni
    if ax == None :
        ax = plt.gca()
        
    # choix du domaine
    if domaine == 'c14' :
        y_name = 'c14age'
        y = IntCal20.c14age.values
        y_sigma = IntCal20.c14sigma.values
    elif domaine == 'f14c' :
        y_name = 'c14age'
        y = c14_to_f14c(IntCal20.c14age.values)
        y_sigma = c14sig_to_f14csig(IntCal20.c14age.values, IntCal20.c14sigma.values)
        IntCal20['c14age'] = y
        IntCal20['c14sigma'] = y_sigma
    else : 
        #domaine = 'delta14c'
        y_name = 'd14c'
        y = IntCal20.d14c.values
        y_sigma = IntCal20.d14csigma.values
    
    # ajout de IntCal20 sur ax
    IntCal20.plot(
        x = 'calage',
        y = y_name,
        label = 'IntCal20',
        figsize = figsize,
        ax = ax,
        color = color
    )
    
    # Incertitude autour de la courbe
    if incertitude :
        ax.fill_between(
            IntCal20.calage.values, 
            y - sigma_length * y_sigma,
            y + sigma_length * y_sigma,
            label=f"IntCal20 $\pm$ {sigma_length} $\sigma$", 
            color= color, 
            alpha=alpha
        )
    
    # setup des axes
    if Min_y != None and Max_y != None :
        ax.set_ylim(Min_y, Max_y)
    if Min_x != None and Max_x != None :
        ax.set_xlim(Min_x, Max_x)
    if invert_xaxis :
        ax.invert_xaxis()
    
    return ax
    
    
    
def plot_IntCal20_curve(
    xlabel: str = "Calibrated dates (years BP)",
    ylabel: str = "Radiocarbon ages (years BP)",
    add_grid: bool = True,
    fontsize_legend: Union[int, str] = 'small',
    reset_margins: bool = False,

    ax: plt.Axes = None,
    figsize: Optional[Tuple[int, int]] = None,
    color: str = "green",
    alpha: float = 0.4,
    incertitude: bool = True,
    sigma_length: int = 1,
    Min_x: Optional[float] = None, 
    Max_x: Optional[float] = None,
    Min_y: Optional[float] = None, 
    Max_y: Optional[float] = None,
    invert_xaxis: bool = True,
    domaine: str = 'delta14c'
):
    """
    Plot the IntCal20 calibration curve in the chosen radiocarbon domain.

    Parameters
    ----------
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label. Adjusted automatically if `domaine` is "delta14c" or "f14c".
    add_grid : bool, optional
        If True, draw a grid behind the plot.
    fontsize_legend : int or str, optional
        Font size passed to matplotlib’s legend.
    reset_margins : bool, optional
        If True, reset axis margins to zero.

    ax : matplotlib.axes.Axes or None
        Existing axis onto which the curve is drawn. If None, a new one is created.
    figsize : tuple(int, int), optional
        Figure size passed to matplotlib.
    color : str, optional
        Color of the IntCal20 curve.
    alpha : float, optional
        Transparency for the uncertainty band.
    incertitude : bool, optional
        If True, display the ±σ IntCal20 uncertainty envelope.
    sigma_length : int, optional
        Number of sigma widths for the uncertainty envelope.

    Min_x, Max_x : float or None, optional
        Optional x-axis limits; when both provided the axis limits are set.
    Min_y, Max_y : float or None, optional
        Optional y-axis limits; when both provided the axis limits are set.
    invert_xaxis : bool, optional
        If True, invert X axis (IntCal standard orientation).
    domaine : {'delta14c', 'c14', 'f14c'}
        Domain in which to express radiocarbon values.

    Returns
    -------
    None
    """

    ax = add_IntCal20_curve(
            ax = ax,
            figsize = figsize,
            color = color,
            alpha= alpha,
            incertitude = incertitude,
            sigma_length = sigma_length,
            Min_x = Min_x, Max_x = Max_x,
            Min_y = Min_y, Max_y = Max_y,
            invert_xaxis = True,
            domaine = domaine
    )

    if reset_margins:
        ax.margins(0)
    
    plt.xlabel(xlabel)
    if domaine == 'delta14c' :
        ylabel = "$\Delta^{14}$C"
    if domaine == 'f14c' :
        ylabel = "F$^{14}$C"
    plt.ylabel(ylabel)
    plt.grid(add_grid)
    #plt.gca().invert_xaxis()
    plt.legend(fontsize=fontsize_legend)
    plt.show()
    
    
# fonction pour trouver beta_opt pour chaque prediction
def find_quantile_beta_opt(
    alpha: float,
    predictions: np.ndarray,
    method: str = 'median_unbiased'
) -> float:
    """
    Compute the optimal quantile-shift parameter ``beta`` that minimizes
    the length of an asymmetric prediction credible interval of 
    level ``1 - alpha``.

    The interval is defined as:
    $$
        [ Q_{\\beta} , Q_{1 - \\alpha + \\beta} ]
    $$
    where $Q_p$ denotes the empirical quantile of the predictive distribution.
    The optimal ``beta`` is obtained by numerical minimization of the
    interval length with respect to ``beta`` over the admissible domain
    [0, alpha].

    Parameters
    ----------
    alpha : float
        Target tail probability. Must satisfy ``0 < alpha < 1``.
    predictions : np.ndarray
        One-dimensional array of predictive samples from which
        empirical quantiles are computed.
    method : str, optional
        Quantile calculation method passed to ``numpy.quantile``.
        Defaults to 'median_unbiased'.

    Returns
    -------
    float
        The optimal ``beta`` in the interval [0, alpha].

    Raises
    ------
    ValueError
        If ``alpha`` is not in (0, 1), if ``predictions`` is not a
        one-dimensional numpy array, or if it contains NaNs.
    """

    # Vérifications d’entrées
    if not isinstance(alpha, (float, int)) or not (0 < float(alpha) < 1):
        raise ValueError("'alpha' must be a float strictly between 0 and 1.")

    if not isinstance(predictions, np.ndarray):
        raise ValueError("'predictions' must be a numpy.ndarray.")

    if predictions.ndim != 1:
        raise ValueError("'predictions' must be one-dimensional.")

    if np.isnan(predictions).any():
        raise ValueError("'predictions' contains NaN values, cannot compute quantiles.")

    interval_length = lambda beta : np.quantile(predictions, 1 - alpha + beta, method=method) - np.quantile(predictions, beta, method=method)
    beta_opt = minimize(fun = interval_length , x0 = np.array([alpha/2]), method = 'Nelder-Mead', bounds = [(0.,alpha)])
    return beta_opt.x[0]

# fonction pour calculer les bornes des intervalles de crédibilité pour les prédictions
# = fonction pour calculer l'intervalle de crédibilité autour de la courbe
def compute_credible_intervals_bounds(
    alpha: float,
    bnn_predictions: np.ndarray,
    method: str = 'median_unbiased'
):
    """
    Compute pointwise (per-prediction) credible-interval bounds for a set of
    predictive samples produced by a Bayesian Neural Network.  

    For each prediction index ``i`` (row of ``bnn_predictions``), the function: 
     
    1. Computes the optimal quantile-shift parameter ``beta_opt`` using
        `find_quantile_beta_opt` to minimize the length of the asymmetric
        credible interval of level ``1 - alpha``.  
    2. Extracts the lower and upper credible bounds using the empirical
        quantiles:  
        - lower = Q_{beta_opt}  
        - upper = Q_{1 - alpha + beta_opt}  

    Parameters
    ----------
    alpha : float
        Target tail probability (the interval has probability mass ``1 - alpha``).  
        Must satisfy ``0 < alpha < 1``.
    bnn_predictions : np.ndarray
        Array of predictive samples, typically of shape
        ``(n_grid_points, n_samples)`` where each row corresponds to the
        distribution of predictions at a given calendar date.
    method : str, optional
        Method used by ``numpy.quantile``. Default is 'median_unbiased'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray], np.ndarray
        A tuple ``(lower_bounds, upper_bounds)`` where both arrays have shape
        ``(n_grid_points,)``, and an array ``beta_opt_list`` of shape
        ``(n_grid_points,)`` containing the optimal beta for each point.

    Note
    -----
    This function is typically used to generate credible bands around 
    reconstructed calibration curves.
    """

    predictions_size = bnn_predictions.shape[0]
    CI_upper_bounds = []
    CI_lower_bounds = []
    beta_opt_list = []
    for i in range(predictions_size) :
        predictions = bnn_predictions[i,]
        beta_opt = find_quantile_beta_opt(alpha, predictions, method=method)
        upper = np.quantile(predictions, 1 - alpha + beta_opt, method=method)
        lower = np.quantile(predictions, beta_opt, method=method)
        CI_upper_bounds.append(upper)
        CI_lower_bounds.append(lower)
        beta_opt_list.append(beta_opt)
    return (np.array(CI_lower_bounds), np.array(CI_upper_bounds)), np.array(beta_opt_list)
    




# courbe BNN


def add_individual_calibration_curve_part_1(
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    color: str = 'cyan',
    alpha: float = .4,
    incertitude: bool = True,
    sigma_length: int = 1,
    Min_x: Optional[float] = None, Max_x: Optional[float] = None,
    Min_y: Optional[float] = None, Max_y: Optional[float] = None,
    invert_xaxis: bool = True,
    domaine: str = ['delta14c', 'c14', 'f14c'][0],
    covariables: bool = False,
    label_prefix: str = '',
    label_suffix: str = '',
    credible_interval: bool = False,
    credible_interval_level: float = 0.95,
    credible_color: Optional[str] = None,
    credible_alpha: Optional[float] = None
) -> plt.Axes:
    """
    Add the Bayesian Neural Network (BNN) calibration curve (part 1) to a Matplotlib axis.

    This function loads precomputed BNN midpoint predictions for the ``part_1`` curve
    (recent segment), optionally converts the predictions into the requested radiocarbon
    domain (`delta14c`, `c14` or `f14c`), computes the pointwise mean and standard
    deviation across Monte-Carlo predictive draws and plots the mean curve.  It can
    optionally display the ±σ band and a pointwise credible interval band (computed
    from predictive samples).

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. If ``None``, the current axis from ``matplotlib.pyplot.gca()``
        is used. Default is ``None``.
    figsize : tuple(int, int) or None, optional
        If provided, forwarded to the plotting call; default is ``None``.
    color : str, optional
        Color used for the BNN mean curve and (by default) for the uncertainty band.
    alpha : float, optional
        Alpha transparency for the ±σ uncertainty band.
    incertitude : bool, optional
        If True, draw the ± `sigma_length` standard-deviation envelope around the mean.
    sigma_length : int, optional
        Number of standard deviations for the ± envelope (e.g. 1 for ±1σ).
    Min_x, Max_x : float or None, optional
        Optional x-axis limits; when both provided the axis limits are set.
    Min_y, Max_y : float or None, optional
        Optional y-axis limits; when both provided the axis limits are set.
    invert_xaxis : bool, optional
        If True, invert the x-axis (IntCal convention). Default True.
    domaine : {'delta14c', 'c14', 'f14c'}, optional
        Domain for the plotted predictions. Default is `'delta14c'`.
    covariables : bool, optional
        Whether to use BNN predictions trained with covariates.
    label_prefix : str, optional
        String prefix to prepend to plotted labels.
    label_suffix : str, optional
        String suffix to append to plotted labels.
    credible_interval : bool, optional
        If True, compute and plot pointwise credible intervals from predictive samples.
    credible_interval_level : float, optional
        Credible interval level (e.g. 0.95 for 95% credible interval). Default 0.95.
    credible_color : str or None, optional
        Color for the credible interval fill. If ``None`` the main ``color`` is used.
    credible_alpha : float or None, optional
        Alpha transparency for the credible interval fill. If ``None`` ``alpha`` is used.

    Returns
    -------
    matplotlib.axes.Axes
        The axis object containing the plotted BNN curve.

    Note
    ----
    Credible intervals are computed pointwise per grid location from the predictive
    samples (no smoothing or joint-region calculation here).

    See Also
    --------
    `compute_credible_intervals_bounds` : helper used to compute pointwise credible bands.

    """
    
    # chargement des prédictions pré-sauvergardées
    
    if covariables :
        filename = "bnn_part_1_with_covariables_middle_points_predictions.csv"
    else :
        filename = "bnn_part_1_without_covariables_middle_points_predictions.csv"
    middle_points_predictions_part_2_filepath = bnn_predictions_dir /  filename
    
    middle_points_predictions_part_2, nb_intervals, nb_curves = bnn_load_predictions_(
        filepath = middle_points_predictions_part_2_filepath
    )
    
    # Paramétrages des bornes pour la partie ancienne de la courbe de calibration
    
    # Bornes des âges dans le dataset d'entraînement de la partie 2
    Max_part_2 = 12310
    Min_part_2 = -4

    # choix bornes sup et inf adaptées à la partie part_2 (courbe partie ancienne) et leur reduction dans [0,1], 
    max_horizon_x = 12310 # horizon temporel des ages calibrés limité à celui de IntCal20
    min_horizon_x_part_2 = -4 # fin de la période avec dates incertaines
    max_horizon_x_part_2_scaled = minimax_scaling(max_horizon_x, Max=Max_part_2, Min=Min_part_2)
    min_horizon_x_part_2_scaled = minimax_scaling(min_horizon_x_part_2, Max=Max_part_2, Min=Min_part_2)

    # Subdivision de l'intervalle de temps couvert par la courbe part_2 en sous intervalles 
    # pour intégration et simulation si nécessaire
    intervals_bounds_part_2 = np.linspace(min_horizon_x_part_2_scaled, max_horizon_x_part_2_scaled, nb_intervals + 1)
    middle_points_part_2 = (intervals_bounds_part_2[1:] + intervals_bounds_part_2[:-1])/2
    
    # dates calibrées en années BP
    calage = minimax_scaling_reciproque(
        x = middle_points_part_2,
        Max = Max_part_2,
        Min = Min_part_2
    )
    
    # Création de l'axe si pas fourni
    if ax == None :
        ax = plt.gca()
        
    # choix du domaine
    if domaine == 'c14' :
        middle_points_predictions_part_2 = d14c_to_c14(
            d14c = middle_points_predictions_part_2, 
            teta = calage.repeat(nb_curves).reshape((nb_intervals, nb_curves))
        )
        y_name = 'c14age'
    elif domaine =='f14c' :
        middle_points_predictions_part_2 = d14c_to_f14c(
            d14c = middle_points_predictions_part_2, 
            teta = calage.repeat(nb_curves).reshape((nb_intervals, nb_curves))
        )
        y_name = 'f14c'
    else : 
        #domaine = 'delta14c'
        y_name = 'd14c'
    
    # calcul courbe moyenne et écart-type
    y = middle_points_predictions_part_2.mean(axis=1)
    y_sigma = middle_points_predictions_part_2.std(axis=1)
    
    # ajout courbe moyenne sur ax
    df = pd.DataFrame({
        'calage': calage,
        y_name: y
    })
    
    df.plot(
        x = 'calage',
        y = y_name,
        label = label_prefix + 'BNN curve' + label_suffix,
        figsize = figsize,
        ax = ax,
        color = color
    )
    
    # Incertitude autour de la courbe
    if incertitude :
        ax.fill_between(
            calage, 
            y - sigma_length * y_sigma,
            y + sigma_length * y_sigma,
            label=label_prefix + "BNN curve" + label_suffix + f"$\pm$ {sigma_length} $\sigma$", 
            color= color, 
            alpha=alpha
        )
        
    # Intervalle de crédibilité
    if credible_interval :
        if credible_alpha == None :
            credible_alpha = alpha
        if credible_color == None :
            credible_color = color
        # calcul intervalle de crédibilité de chaque prédiction
        credible_intervals, _ = compute_credible_intervals_bounds(
            1 - credible_interval_level, middle_points_predictions_part_2
        )
        ax.fill_between(
            calage, 
            credible_intervals[0], 
            credible_intervals[1], 
            label=f"{round(100*credible_interval_level)}% CI for BNN curve", 
            color=credible_color, 
            alpha=credible_alpha
        )
        
        
        
    # setup des axes
    if Min_y != None and Max_y != None :
        ax.set_ylim(Min_y, Max_y)
    if Min_x != None and Max_x != None :
        ax.set_xlim(Min_x, Max_x)
    if invert_xaxis :
        ax.invert_xaxis()
    
    return ax


def plot_individual_calibration_curve_part_1(
    xlabel: str = "Calibrated dates (in years BP)",
    ylabel: str = "Radiocarbon ages (in years BP)",
    add_grid: bool = True,
    fontsize_legend: str = 'small',
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    color: str = 'cyan',
    alpha: float = .4,
    incertitude: bool = True,
    sigma_length: int = 1,
    Min_x: Optional[float] = None, Max_x: Optional[float] = None,
    Min_y: Optional[float] = None, Max_y: Optional[float] = None,
    invert_xaxis: bool = True,
    domaine: str = ['delta14c', 'c14', 'f14c'][0],
    covariables: bool = False,
    credible_interval: bool = False,
    credible_interval_level: float = 0.95,
    credible_color: Optional[str] = None,
    credible_alpha: Optional[float] = None
) -> None:
    """
    Plot the BNN calibration curve (part 1) by calling
    ``add_individual_calibration_curve_part_1`` and applying axis labels,
    grid, legend, and final display.

    This function is a thin wrapper around
    ``add_individual_calibration_curve_part_1`` and only handles visual
    elements such as xlabel, ylabel, grid and legend.

    Parameters
    ----------
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis. Adjusted automatically if `domaine` is `"delta14c"` or `"f14c"`.
    add_grid : bool, optional
        Whether to display a grid on the plot.
    fontsize_legend : str, optional
        Font size for the legend.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. If ``None``, the axis internally created by
        ``add_individual_calibration_curve_part_1`` is used.
    figsize : tuple(int, int) or None, optional
        Figure size forwarded to the plotting function.
    color : str, optional
        Color used for plotting the BNN mean curve.
    alpha : float, optional
        Transparency for uncertainty or credible interval fills.
    incertitude : bool, optional
        Whether to draw the ±σ uncertainty band around the mean curve.
    sigma_length : int, optional
        Number of standard deviations for the uncertainty band.
    Min_x, Max_x : float or None, optional
        Optional x-axis limits; when both provided the axis limits are set.
    Min_y, Max_y : float or None, optional
        Optional y-axis limits; when both provided the axis limits are set.
    invert_xaxis : bool, optional
        If True, the x-axis is inverted (IntCal convention).
    domaine : {'delta14c', 'c14', 'f14c'}, optional
        Domain of radiocarbon quantities displayed.
    covariables : bool, optional
        Whether to use BNN predictions trained with covariates.
    credible_interval : bool, optional
        If True, compute and plot pointwise credible intervals from predictive samples.
    credible_interval_level : float, optional
        Credible interval level (e.g. 0.95 for 95% credible interval). Default 0.95.
    credible_color : str or None, optional
        Color for the credible interval fill. If ``None`` the main ``color`` is used.
    credible_alpha : float or None, optional
        Alpha transparency for the credible interval fill. If ``None`` ``alpha`` is used.

    Returns
    -------
    None
        The function produces a plot.
    """
    ax = add_individual_calibration_curve_part_1(
            ax = ax,
            figsize = figsize,
            color = color,
            alpha= alpha,
            incertitude = incertitude,
            sigma_length = sigma_length,
            Min_x = Min_x, Max_x = Max_x,
            Min_y = Min_y, Max_y = Max_y,
            invert_xaxis = True,
            domaine = domaine,
            covariables = covariables,
            credible_interval = credible_interval,
            credible_interval_level = credible_interval_level,
            credible_color = credible_color,
            credible_alpha = credible_alpha
    )
    
    plt.xlabel(xlabel)
    if domaine == 'delta14c' :
        ylabel = "$\Delta^{14}$C"
    if domaine == 'f14c' :
        ylabel = "F$^{14}$C"
    plt.ylabel(ylabel)
    plt.grid(add_grid)
    plt.legend(fontsize=fontsize_legend)
    plt.show()


def add_individual_calibration_curve_part_2(
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    color: str = 'cyan',
    alpha: float = .4,
    incertitude: bool = True,
    sigma_length: int = 1,
    Min_x: Optional[float] = None, Max_x: Optional[float] = None,
    Min_y: Optional[float] = None, Max_y: Optional[float] = None,
    invert_xaxis: bool = True,
    domaine: str = ['delta14c', 'c14', 'f14c'][0],
    covariables: bool = False,
    label_prefix: str = '',
    label_suffix: str = '',
    credible_interval: bool = False,
    credible_interval_level: float = 0.95,
    credible_color: Optional[str] = None,
    credible_alpha: Optional[float] = None
) -> plt.Axes:
    """
    Add the Bayesian Neural Network (BNN) calibration curve (part 2) to a Matplotlib axis.

    This function loads precomputed BNN midpoint predictions for the ``part_2`` curve
    (old segment), optionally converts the predictions into the requested radiocarbon
    domain (`delta14c`, `c14` or `f14c`), computes the pointwise mean and standard
    deviation across Monte-Carlo predictive draws and plots the mean curve.  It can
    optionally display the ±σ band and a pointwise credible interval band (computed
    from predictive samples).

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. If ``None``, the current axis from ``matplotlib.pyplot.gca()``
        is used. Default is ``None``.
    figsize : tuple(int, int) or None, optional
        If provided, forwarded to the plotting call; default is ``None``.
    color : str, optional
        Color used for the BNN mean curve and (by default) for the uncertainty band.
    alpha : float, optional
        Alpha transparency for the ±σ uncertainty band.
    incertitude : bool, optional
        If True, draw the ± `sigma_length` standard-deviation envelope around the mean.
    sigma_length : int, optional
        Number of standard deviations for the ± envelope (e.g. 1 for ±1σ).
    Min_x, Max_x : float or None, optional
        Optional x-axis limits; when both provided the axis limits are set.
    Min_y, Max_y : float or None, optional
        Optional y-axis limits; when both provided the axis limits are set.
    invert_xaxis : bool, optional
        If True, invert the x-axis (IntCal convention). Default True.
    domaine : {'delta14c', 'c14', 'f14c'}, optional
        Domain for the plotted predictions. Default is `'delta14c'`.
    covariables : bool, optional
        Whether to use BNN predictions trained with covariates.
    label_prefix : str, optional
        String prefix to prepend to plotted labels.
    label_suffix : str, optional
        String suffix to append to plotted labels.
    credible_interval : bool, optional
        If True, compute and plot pointwise credible intervals from predictive samples.
    credible_interval_level : float, optional
        Credible interval level (e.g. 0.95 for 95% credible interval). Default 0.95.
    credible_color : str or None, optional
        Color for the credible interval fill. If ``None`` the main ``color`` is used.
    credible_alpha : float or None, optional
        Alpha transparency for the credible interval fill. If ``None`` ``alpha`` is used.

    Returns
    -------
    matplotlib.axes.Axes
        The axis object containing the plotted BNN curve.

    Note
    ----
    Credible intervals are computed pointwise per grid location from the predictive
    samples (no smoothing or joint-region calculation here).

    See Also
    --------
    `compute_credible_intervals_bounds` : helper used to compute pointwise credible bands.

    """
    
    # chargement des prédictions pré-sauvergardées
    
    if covariables :
        filename = "bnn_part_2_with_covariables_middle_points_predictions.csv"
    else :
        filename = "bnn_part_2_without_covariables_middle_points_predictions.csv"
    middle_points_predictions_part_2_filepath = bnn_predictions_dir /  filename
    
    middle_points_predictions_part_2, nb_intervals, nb_curves = bnn_load_predictions_(
        filepath = middle_points_predictions_part_2_filepath
    )
    
    # Paramétrages des bornes pour la partie ancienne de la courbe de calibration
    
    # Bornes des âges dans le dataset d'entraînement de la partie 2
    Max_part_2 = 63648
    Min_part_2 = 12283

    # choix bornes sup et inf adaptées à la partie part_2 (courbe partie ancienne) et leur reduction dans [0,1], 
    max_horizon_x = 55000 # horizon temporel des ages calibrés limité à celui de IntCal20
    min_horizon_x_part_2 = 12310 # fin de la période avec dates incertaines
    max_horizon_x_part_2_scaled = minimax_scaling(max_horizon_x, Max=Max_part_2, Min=Min_part_2)
    min_horizon_x_part_2_scaled = minimax_scaling(min_horizon_x_part_2, Max=Max_part_2, Min=Min_part_2)

    # Subdivision de l'intervalle de temps couvert par la courbe part_2 en sous intervalles 
    # pour intégration et simulation si nécessaire
    intervals_bounds_part_2 = np.linspace(min_horizon_x_part_2_scaled, max_horizon_x_part_2_scaled, nb_intervals + 1)
    middle_points_part_2 = (intervals_bounds_part_2[1:] + intervals_bounds_part_2[:-1])/2
    
    # dates calibrées en années BP
    calage = minimax_scaling_reciproque(
        x = middle_points_part_2,
        Max = Max_part_2,
        Min = Min_part_2
    )
    
    # Création de l'axe si pas fourni
    if ax == None :
        ax = plt.gca()
        
    # choix du domaine
    if domaine == 'c14' :
        middle_points_predictions_part_2 = d14c_to_c14(
            d14c = middle_points_predictions_part_2, 
            teta = calage.repeat(nb_curves).reshape((nb_intervals, nb_curves))
        )
        y_name = 'c14age'
    elif domaine =='f14c' :
        middle_points_predictions_part_2 = d14c_to_f14c(
            d14c = middle_points_predictions_part_2, 
            teta = calage.repeat(nb_curves).reshape((nb_intervals, nb_curves))
        )
        y_name = 'f14c'
    else : 
        #domaine = 'delta14c'
        y_name = 'd14c'
    
    # calcul courbe moyenne et écart-type
    y = middle_points_predictions_part_2.mean(axis=1)
    y_sigma = middle_points_predictions_part_2.std(axis=1)
    
    # ajout courbe moyenne sur ax
    df = pd.DataFrame({
        'calage': calage,
        y_name: y
    })
    
    df.plot(
        x = 'calage',
        y = y_name,
        label = label_prefix + 'BNN curve' + label_suffix,
        figsize = figsize,
        ax = ax,
        color = color
    )
    
    # Incertitude autour de la courbe
    if incertitude :
        ax.fill_between(
            calage, 
            y - sigma_length * y_sigma,
            y + sigma_length * y_sigma,
            label=label_prefix + "BNN curve" + label_suffix + f"$\pm$ {sigma_length} $\sigma$", 
            color= color, 
            alpha=alpha
        )
        
    # Intervalle de crédibilité
    if credible_interval :
        if credible_alpha == None :
            credible_alpha = alpha
        if credible_color == None :
            credible_color = color
        # calcul intervalle de crédibilité de chaque prédiction
        credible_intervals, _ = compute_credible_intervals_bounds(
            1 - credible_interval_level, middle_points_predictions_part_2
        )
        ax.fill_between(
            calage, 
            credible_intervals[0], 
            credible_intervals[1], 
            label=f"{round(100*credible_interval_level)}% CI for BNN curve", 
            color=credible_color, 
            alpha=credible_alpha
        )
        
        
        
    # setup des axes
    if Min_y != None and Max_y != None :
        ax.set_ylim(Min_y, Max_y)
    if Min_x != None and Max_x != None :
        ax.set_xlim(Min_x, Max_x)
    if invert_xaxis :
        ax.invert_xaxis()
    
    return ax


def plot_individual_calibration_curve_part_2(
    xlabel: str = "Calibrated dates (in years BP)",
    ylabel: str = "Radiocarbon ages (in years BP)",
    add_grid: bool = True,
    fontsize_legend: str = 'small',
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    color: str = 'cyan',
    alpha: float = .4,
    incertitude: bool = True,
    sigma_length: int = 1,
    Min_x: Optional[float] = None, Max_x: Optional[float] = None,
    Min_y: Optional[float] = None, Max_y: Optional[float] = None,
    invert_xaxis: bool = True,
    domaine: str = ['delta14c', 'c14', 'f14c'][0],
    covariables: bool = False,
    credible_interval: bool = False,
    credible_interval_level: float = 0.95,
    credible_color: Optional[str] = None,
    credible_alpha: Optional[float] = None
) -> None:
    """
    Plot the BNN calibration curve (part 2) by calling
    ``add_individual_calibration_curve_part_2`` and applying axis labels,
    grid, legend, and final display.

    This function is a thin wrapper around
    ``add_individual_calibration_curve_part_2`` and only handles visual
    elements such as xlabel, ylabel, grid and legend.

    Parameters
    ----------
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis. Adjusted automatically if `domaine` is `"delta14c"` or `"f14c"`.
    add_grid : bool, optional
        Whether to display a grid on the plot.
    fontsize_legend : str, optional
        Font size for the legend.
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. If ``None``, the axis internally created by
        ``add_individual_calibration_curve_part_2`` is used.
    figsize : tuple(int, int) or None, optional
        Figure size forwarded to the plotting function.
    color : str, optional
        Color used for plotting the BNN mean curve.
    alpha : float, optional
        Transparency for uncertainty or credible interval fills.
    incertitude : bool, optional
        Whether to draw the ±σ uncertainty band around the mean curve.
    sigma_length : int, optional
        Number of standard deviations for the uncertainty band.
    Min_x, Max_x : float or None, optional
        Optional x-axis limits; when both provided the axis limits are set.
    Min_y, Max_y : float or None, optional
        Optional y-axis limits; when both provided the axis limits are set.
    invert_xaxis : bool, optional
        If True, the x-axis is inverted (IntCal convention).
    domaine : {'delta14c', 'c14', 'f14c'}, optional
        Domain of radiocarbon quantities displayed.
    covariables : bool, optional
        Whether to use BNN predictions trained with covariates.
    credible_interval : bool, optional
        If True, compute and plot pointwise credible intervals from predictive samples.
    credible_interval_level : float, optional
        Credible interval level (e.g. 0.95 for 95% credible interval). Default 0.95.
    credible_color : str or None, optional
        Color for the credible interval fill. If ``None`` the main ``color`` is used.
    credible_alpha : float or None, optional
        Alpha transparency for the credible interval fill. If ``None`` ``alpha`` is used.

    Returns
    -------
    None
        The function produces a plot.
    """
    ax = add_individual_calibration_curve_part_2(
            ax = ax,
            figsize = figsize,
            color = color,
            alpha= alpha,
            incertitude = incertitude,
            sigma_length = sigma_length,
            Min_x = Min_x, Max_x = Max_x,
            Min_y = Min_y, Max_y = Max_y,
            invert_xaxis = True,
            domaine = domaine,
            covariables = covariables,
            credible_interval = credible_interval,
            credible_interval_level = credible_interval_level,
            credible_color = credible_color,
            credible_alpha = credible_alpha
    )
    
    plt.xlabel(xlabel)
    if domaine == 'delta14c' :
        ylabel = "$\Delta^{14}$C"
    if domaine == 'f14c' :
        ylabel = "F$^{14}$C"
    plt.ylabel(ylabel)
    plt.grid(add_grid)
    plt.legend(fontsize=fontsize_legend)
    plt.show()
    

def add_individual_calibration_curve_parts_1_and_2(
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    color: str = 'cyan',
    alpha: float = .4,
    incertitude: bool = True,
    sigma_length: int = 1,
    Min_x: Optional[float] = None, Max_x: Optional[float] = None,
    Min_y: Optional[float] = None, Max_y: Optional[float] = None,
    invert_xaxis: bool = True,
    domaine: str = ['delta14c', 'c14', 'f14c'][0],
    covariables: bool = False,
    credible_interval: bool = False,
    credible_interval_level: float = 0.95,
    credible_color: Optional[str] = None,
    credible_alpha: Optional[float] = None
) -> plt.Axes:
    """
    Add both parts (1 and 2) of the BNN calibration curve to the
    provided matplotlib axis.

    This function successively calls:

    - ``add_individual_calibration_curve_part_1`` for the recent period,
    - ``add_individual_calibration_curve_part_2`` for the older period.

    It manages:
    
    - axis inversion only once,
    - legend suffix depending on whether covariates are used,
    - forwarding of all visual and uncertainty options,
    - consistent handling of credible intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, optional
        Axis on which to draw the curves. If ``None``, the axis is taken from
        ``add_individual_calibration_curve_part_1``.
    figsize : tuple(int, int) or None, optional
        Figure size used by the underlying plot calls.
    color : str, optional
        Color used for both BNN curve parts.
    alpha : float, optional
        Transparency for uncertainty or credible interval fills.
    incertitude : bool, optional
        Whether to draw the ±σ uncertainty band around the mean curve.
    sigma_length : int, optional
        Number of standard deviations for the uncertainty band.
    Min_x, Max_x : float or None, optional
        Optional x-axis limits; when both provided the axis limits are set.
    Min_y, Max_y : float or None, optional
        Optional y-axis limits; when both provided the axis limits are set.
    invert_xaxis : bool, optional
        If True, the x-axis is inverted (IntCal convention).
    domaine : {'delta14c', 'c14', 'f14c'}, optional
        Domain of radiocarbon quantities displayed.
    covariables : bool, optional
        Whether to use BNN predictions trained with covariates.
        This affects the legend suffix.
    credible_interval : bool, optional
        If True, compute and plot pointwise credible intervals from predictive samples.
    credible_interval_level : float, optional
        Credible interval level (e.g. 0.95 for 95% credible interval). Default 0.95.
    credible_color : str or None, optional
        Color for the credible interval fill. If ``None`` the main ``color`` is used.
    credible_alpha : float or None, optional
        Alpha transparency for the credible interval fill. If ``None`` ``alpha`` is used.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing both parts of the BNN calibration curve.
    """

    if not covariables :
        # à donner uniquement à un graphique (ici part_1) pour la légende en cas de covariables
        label_suffix = ' with covariates '
    else :
        label_suffix = ''

    
    # ajout partie de la courbe sans dates incertaines
    ax = add_individual_calibration_curve_part_1(
            ax = ax,
            figsize = figsize,
            color = color,
            alpha= alpha,
            incertitude = incertitude,
            sigma_length = sigma_length,
            Min_x = Min_x, Max_x = Max_x,
            Min_y = Min_y, Max_y = Max_y,
            invert_xaxis = invert_xaxis,
            domaine = domaine,
            covariables = covariables,
            label_prefix = '', # la légende apparaît
            label_suffix = label_suffix, # gestion de la présence ou non des covariables dans la légende
            credible_interval = credible_interval,
            credible_interval_level = credible_interval_level,
            credible_color = credible_color,
            credible_alpha = credible_alpha
    )

    if invert_xaxis :
        # on n'inverse plus l'axe une deuxième fois
        invert_xaxis = False
    
    ax = add_individual_calibration_curve_part_2(
            ax = ax,
            figsize = figsize,
            color = color,
            alpha= alpha,
            incertitude = incertitude,
            sigma_length = sigma_length,
            Min_x = Min_x, Max_x = Max_x,
            Min_y = Min_y, Max_y = Max_y,
            invert_xaxis = invert_xaxis,
            domaine = domaine,
            covariables = covariables,
            label_prefix = '_', # pour supprimer la double apparition dans la légende
            label_suffix = '',
            credible_interval = credible_interval,
            credible_interval_level = credible_interval_level,
            credible_color = credible_color,
            credible_alpha = credible_alpha
    )
    
    return ax


def add_bnn_calibration_curve(
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    color: str = 'cyan',
    alpha: float = .4,
    incertitude: bool = True,
    sigma_length: int = 1,
    Min_x: Optional[float] = None, Max_x: Optional[float] = None,
    Min_y: Optional[float] = None, Max_y: Optional[float] = None,
    invert_xaxis: bool = True,
    domaine: str = ['delta14c', 'c14', 'f14c'][0],
    covariables: bool = False,
    label_prefix: str = '',
    label_suffix: str = '',
    credible_interval: bool = False,
    credible_interval_level: float = 0.95,
    credible_color: Optional[str] = None,
    credible_alpha: Optional[float] = None
) -> plt.Axes:
    """
    Add the full BNN calibration curve (concatenated parts 1 and 2) to a given
    matplotlib axis.

    This function:
    
    - loads precomputed BNN predictions from both sections of the calibration model,
      previously concatenated using ``concatenate_curves_parts``;
    - rescales the middle points to calendar ages (BP);
    - converts predictions to the selected domain (Δ14C, 14C age, or F14C);
    - plots the mean BNN prediction curve;
    - optionally adds the ±σ uncertainty band;
    - optionally adds credible intervals obtained from the posterior predictive samples.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or None, optional
        Axis to draw on. If ``None``, the current axis from ``matplotlib.pyplot.gca()``
        is used. Default is ``None``.
    figsize : tuple(int, int) or None, optional
        Size of the matplotlib figure used by the underlying ``DataFrame.plot`` call.
    color : str, optional
        Color used for plotting the curve and uncertainty bands.
    alpha : float, optional
        Transparency level for uncertainty shading and credible interval bands.
    incertitude : bool, optional
        Whether to add the ±σ uncertainty band around the mean curve.
    sigma_length : int, optional
        Width of the uncertainty band in standard deviations.
    Min_x, Max_x : float or None, optional
        Optional x-axis limits; when both provided the axis limits are set.
    Min_y, Max_y : float or None, optional
        Optional y-axis limits; when both provided the axis limits are set.
    invert_xaxis : bool, optional
        Whether to invert the x-axis (used for IntCal-style plots).
    domaine : {'delta14c', 'c14', 'f14c'}, optional
        Domain of radiocarbon quantities displayed.
    covariables : bool, optional
        Whether to use BNN predictions trained with covariates.
    label_prefix : str, optional
        String prepended to the curve label.
    label_suffix : str, optional
        String appended to the curve label.
    credible_interval : bool, optional
        Whether to compute and display credible intervals for the BNN predictions.
    credible_interval_level : float, optional
        Credible interval level (e.g., 0.95 for 95% CI).
    credible_color : str or None, optional
        Color of the credible interval shading (defaults to ``color``).
    credible_alpha : float or None, optional
        Transparency of the credible interval shading (defaults to ``alpha``).

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plotted BNN calibration curve.
    """

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
    
    # intervals_bounds est une liste contenant les bornes des intervalles de 2 parties de la courbe
    # avec  borne max part 1 = borne min part 2 repétée donc 2 fois. 
    # nb_intervals = nombre de bornes - 2 et pas -1
    nb_intervals = len(intervals_bounds) - 2 
    
    # dates calibrées en années BP
    calage = minimax_scaling_reciproque(
        x = middle_points,
        Max = Max_part_2,
        Min = Min_part_1
    )
    
    # Création de l'axe si pas fourni
    if ax == None :
        ax = plt.gca()
        
    # choix du domaine
    if domaine == 'c14' :
        middle_points_predictions = f14c_to_c14(
            f14c = middle_points_predictions
        )
        y_name = 'c14age'
    elif domaine =='f14c' :
        y_name = 'f14c'
    else : 
        #domaine = 'delta14c'
        middle_points_predictions = f14c_to_d14c(
            f14c = middle_points_predictions, 
            teta = calage.repeat(nb_curves).reshape((nb_intervals, nb_curves))
        )
        y_name = 'd14c'
    
    # calcul courbe moyenne et écart-type
    y = middle_points_predictions.mean(axis=1)
    y_sigma = middle_points_predictions.std(axis=1)
    
    # ajout courbe moyenne sur ax
    df = pd.DataFrame({
        'calage': calage,
        y_name: y
    })
    
    df.plot(
        x = 'calage',
        y = y_name,
        label = label_prefix + 'BNN curve' + label_suffix,
        figsize = figsize,
        ax = ax,
        color = color
    )
    
    # Incertitude autour de la courbe
    if incertitude :
        ax.fill_between(
            calage, 
            y - sigma_length * y_sigma,
            y + sigma_length * y_sigma,
            label=label_prefix + "BNN curve" + label_suffix + f"$\pm$ {sigma_length} $\sigma$", 
            color= color, 
            alpha=alpha
        )
        
    # Intervalle de crédibilité
    if credible_interval :
        if credible_alpha == None :
            credible_alpha = alpha
        if credible_color == None :
            credible_color = color
        # calcul intervalle de crédibilité de chaque prédiction
        credible_intervals, _ = compute_credible_intervals_bounds(
            1 - credible_interval_level, middle_points_predictions
        )
        ax.fill_between(
            calage, 
            credible_intervals[0], 
            credible_intervals[1], 
            label=f"{round(100*credible_interval_level)}% CI for BNN curve", 
            color=credible_color, 
            alpha=credible_alpha
        )
        
        
        
    # setup des axes
    if Min_y != None and Max_y != None :
        ax.set_ylim(Min_y, Max_y)
    if Min_x != None and Max_x != None :
        ax.set_xlim(Min_x, Max_x)
    if invert_xaxis :
        ax.invert_xaxis()
    
    return ax



def plot_bnn_calibration_curve(
    xlabel: str = "Calendar dates (in years BP)",
    ylabel: str = "Radiocarbon ages (in years BP)",
    add_grid: bool = True,
    fontsize_legend: str = 'small',
    reset_margins: bool = False,

    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[int, int]] = None,
    color: str = 'cyan',
    alpha: float = .4,
    incertitude: bool = True,
    sigma_length: float = 1,
    Min_x: Optional[float] = None, Max_x: Optional[float] = None,
    Min_y: Optional[float] = None, Max_y: Optional[float] = None,
    invert_xaxis: bool = True,
    domaine: str = ['delta14c', 'c14', 'f14c'][0],
    covariables: bool = False,
    credible_interval: bool = False,
    credible_interval_level: float = 0.95,
    credible_color: Optional[str] = None,
    credible_alpha: Optional[float] = None
) -> None:
    """
    Plot the full BNN-based calibration curve, including optional uncertainty
    visualization, credible intervals, grid, labels, and legend. This function
    acts as a convenience wrapper around `add_bnn_calibration_curve`, and 
    automatically configures axis labels, grid display, margins, and legend 
    formatting.

    Parameters
    ----------
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label. Adjusted automatically if `domaine` is "delta14c" or "f14c".
    add_grid : bool, optional
        If True, draw a grid behind the plot.
    fontsize_legend : str, optional
        Legend font size (e.g., "small", "medium", "large").
    reset_margins : bool, optional
        Whether to remove extra margins around the plotted data.
    ax : matplotlib.axes.Axes or None, optional
        Axis on which to draw the curves. If ``None``, the axis is taken from
        `add_bnn_calibration_curve`.
    figsize : tuple or None
        Figure size passed to underlying plotting functions.
    color : str, optional
        Color used for plotting the curve and uncertainty bands.
    alpha : float, optional
        Transparency level for uncertainty shading and credible interval bands.
    incertitude : bool, optional
        Whether to add the ±σ uncertainty band around the mean curve.
    sigma_length : float, optional
        Width of the uncertainty band in standard deviations.
    Min_x, Max_x : float or None, optional
        Optional x-axis limits; when both provided the axis limits are set.
    Min_y, Max_y : float or None, optional
        Optional y-axis limits; when both provided the axis limits are set.
    invert_xaxis : bool, optional
        Whether to invert the x-axis (used for IntCal-style plots).
    domaine : {'delta14c', 'c14', 'f14c'}, optional
        Domain of radiocarbon quantities displayed.
    covariables : bool, optional
        Whether to use BNN predictions trained with covariates; 
        influences legend content.
    credible_interval : bool, optional
        Whether to compute and display credible intervals for the BNN predictions.
    credible_interval_level : float, optional
        Credible interval level (e.g., 0.95 for 95% CI).
    credible_color : str or None, optional
        Color of the credible interval shading (defaults to ``color``).
    credible_alpha : float or None, optional
        Transparency of the credible interval shading (defaults to ``alpha``).

    Returns
    -------
    None
        The function produces a plot.
    """
    ax = add_bnn_calibration_curve(
            ax = ax,
            figsize = figsize,
            color = color,
            alpha= alpha,
            incertitude = incertitude,
            sigma_length = sigma_length,
            Min_x = Min_x, Max_x = Max_x,
            Min_y = Min_y, Max_y = Max_y,
            invert_xaxis = True,
            domaine = domaine,
            covariables = covariables,
            credible_interval = credible_interval,
            credible_interval_level = credible_interval_level,
            credible_color = credible_color,
            credible_alpha = credible_alpha
    )

    if reset_margins:
        ax.margins(0)
    
    plt.xlabel(xlabel)
    if domaine == 'delta14c' :
        ylabel = "$\Delta^{14}$C"
    if domaine == 'f14c' :
        ylabel = "F$^{14}$C"
    plt.ylabel(ylabel)
    plt.grid(add_grid)
    plt.legend(fontsize=fontsize_legend)
    plt.show()

# =============================================================================
# Résultats de calibration individuelle
# =============================================================================



# représentation graphique densité date calibrée et région HPD
def add_cal_date_density_plot_and_HPD_region(
    calibration_results: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    color: str = "cyan",
    eps: float = 10**(-7),
    set_title: bool = True,
    add_legend: bool = True,
    plot_HPD_bounds: bool = False,
    plot_HPD_threshold: bool = False
) -> plt.Axes:
    """
    Plot the posterior density of a calibrated date together with its HPD (Highest 
    Posterior Density) region. This function visualizes the posterior distribution 
    computed from calibration results and highlights all HPD intervals by shading 
    and optional bound/threshold markers.

    Parameters
    ----------
    calibration_results : Dict[str, Any]
        Dictionary containing all outputs of the calibration process, including:

        - `'middle_points'` : 1D array of time grid points.
        - `'middle_points_density'` : posterior density evaluated at each grid point.
        - `'alpha'` : tail probability used for HPD region computation (e.g. 0.05 for 95% HPD).
        - `'HPD_threshold'` : density threshold defining the HPD region.
        - `'connexe_HPD_intervals_unscaled_round'` : list of HPD intervals (tuples).
    ax : matplotlib.axes.Axes or None, optional
        Axis on which to draw. If None, the current axis (`plt.gca()`) is used.
    color : str, optional
        Color used for curves and shaded HPD region.
    eps : float, optional
        Minimal normalized density threshold; values below this are considered zero 
        for plotting purposes.
    set_title : bool, optional
        Whether the figure title should be added.
    add_legend : bool, optional
        Whether to display the legend.
    plot_HPD_bounds : bool, optional
        Whether to plot vertical dashed lines at HPD interval bounds.
    plot_HPD_threshold : bool, optional
        Whether to plot the threshold line used to compute the HPD region.

    Returns
    -------
    matplotlib.axes.Axes
        The axis where the plot was added.
    """
    #if type(ax) == 'NoneType' :
    if ax == None :
        ax = plt.gca()
    
    middle_points = calibration_results['middle_points']
    middle_points_density = calibration_results['middle_points_density']
    
    ages_idx=np.where(middle_points_density/middle_points_density.sum()>eps)[0]
    Min_age = middle_points[ages_idx].min()
    Max_age = middle_points[ages_idx].max()
    
    # niveau des intervalles HPD
    alpha = calibration_results['alpha']
    
    #distribution a posteriori sur les dates (date calibrée)
    ax.plot(
        middle_points, 
        middle_points_density/middle_points_density.sum(), 
        label="posterior density of\nthe calibrated date",
        color=color
    )
    
    # intervales HPD
    
    HPD_threshold = calibration_results['HPD_threshold']
    if plot_HPD_threshold :
        ax.plot(
            middle_points, 
            [HPD_threshold]*len(middle_points), 
            '-.', 
            color='gray', 
            #label=f"Seuil utilisé pour déterminer les intervalles HPD de niveau {int((1-alpha)*100)}%")
            label=f"the threshold used to compute the {int((1-alpha)*100)}% HPD region")
    

    
    connexe_HPD_intervals = calibration_results['connexe_HPD_intervals_unscaled_round']
    nb_HPDI = len(connexe_HPD_intervals)
    
    for i in range(nb_HPDI-1) :
        # borne inf de l'intervalle i
        if plot_HPD_bounds :
            ax.plot(
                [connexe_HPD_intervals[i][0],connexe_HPD_intervals[i][0]],
                [0,HPD_threshold], 
                '--', 
                color='red'
            ) 

            # borne sup de l'intervalle i
            ax.plot(
                [connexe_HPD_intervals[i][1],connexe_HPD_intervals[i][1]],
                [0,HPD_threshold], 
                '--', 
                color='red'
            ) 
        
        idx_i = np.where(
            (middle_points >= connexe_HPD_intervals[i][0])*(middle_points <= connexe_HPD_intervals[i][1])
        )[0]
        ax.fill_between(
            middle_points[idx_i],
            0,
            middle_points_density[idx_i]/middle_points_density.sum(),
            color=color, 
            alpha=.3
        )

    # dernier intervalle traité à part pour mettre la légende (le label) une seule fois :
    
    if plot_HPD_bounds:
        # borne inf du dernier intervalle 
        ax.plot(
            [connexe_HPD_intervals[nb_HPDI-1][0],connexe_HPD_intervals[nb_HPDI-1][0]],
            [0,HPD_threshold], 
            '--', 
            color='red'
        )

        # borne sup du dernier intervalle
        ax.plot(
            [connexe_HPD_intervals[nb_HPDI-1][1],connexe_HPD_intervals[nb_HPDI-1][1]],
            [0,HPD_threshold], 
            '--', 
            color='red', 
            #label=f"Borne(s) des/d'intervalle(s) HPD de niveau {int((1-alpha)*100)}%"
            label="HPD region bounds"
        )
    
    idx_nb_HPDI = np.where(
        (middle_points >= connexe_HPD_intervals[nb_HPDI-1][0])*(middle_points <= connexe_HPD_intervals[nb_HPDI-1][1])
    )[0]
    ax.fill_between(
        middle_points[idx_nb_HPDI], 
        0, 
        middle_points_density[idx_nb_HPDI]/middle_points_density.sum(), 
        color=color, 
        alpha=.3, 
        #label=f"intervalle(s) HPD de niveau {int((1-alpha)*100)}%"
        label=f"{int((1-alpha)*100)}% HPD region"
    )

    
    if set_title :
        ax.set_title('HPD region and Posterior distribution\nof the calibrated age')
    ax.set_ylabel('Probability')
    ax.set_xlabel('calibrated dates (in years cal BP)')
    ax.set_xlim(Min_age, Max_age)
    ax.invert_xaxis()
    
    if add_legend :
        ax.legend(loc='lower center', fancybox=True, framealpha=0., bbox_to_anchor=(0.5,-.4))
    
    return ax





# représentation graphique densité âge c14
def add_c14age_density_plot(
    c14age: float,
    c14sig: float,
    ax: Optional[plt.Axes] = None,
    color: str = "gray",
    support_size: float = 5,
    sample_size: int = 1000,
    plot_density: bool = False,
    fill_density: bool = True
) -> plt.Axes:
    """
    Plot the probability density function of a radiocarbon (14C) age measurement,
    modeled as a Gaussian distribution with mean `c14age` and standard deviation
    `c14sig`. The plot can display the curve itself, a filled area, or both.

    Parameters
    ----------
    c14age : float
        The measured conventional radiocarbon age (in years BP).
    c14sig : float
        The standard deviation (1σ) of the radiocarbon age measurement.
    ax : matplotlib.axes.Axes or None, optional
        Axis on which the plot is drawn. If None, the current axis (`plt.gca()`) is used.
    color : str, optional
        Color used for drawing or filling the density.
    support_size : float, optional
        Half-width (in units of standard deviations) of the interval over which
        the density is evaluated.
    sample_size : int, optional
        Number of sample points used for plotting the density curve.
    plot_density : bool, optional
        If True, the density curve is drawn.
    fill_density : bool, optional
        If True, the area between the y-axis and the density curve is filled.

    Returns
    -------
    matplotlib.axes.Axes
        The axis on which the plot is added.

    Note
    ----
    The distribution is actually Gaussian when radiocarbon measurements are expressed 
    in terms of F$^{14}$C values. Although conventional radiocarbon ages are no longer 
    Gaussian, Gaussian density is still used solely for graphical representation purposes.
    """
    #if type(ax) == 'NoneType' :
    if ax == None :
        ax = plt.gca()
    
    
    # densité de la mesure d'âge c14
    mesure_density_fct = lambda m : np.exp(-(m - c14age)**2/(2*c14sig**2))/(c14sig*np.sqrt(2*np.pi))
    mesures = np.linspace(c14age - support_size*c14sig, c14age + support_size*c14sig, sample_size)
    mesures_density = mesure_density_fct(mesures)
    
    # plot de la distribution de la mesure de laboratoire
    label="distribution of $^{14}$C age" +f"\nof {int(c14age)} $\pm$ {round(c14sig, 2)} years BP"
    if plot_density :
        ax.plot(
            mesures_density/mesures_density.sum(), 
            mesures, 
            color=color,
            label=label
        )
    if fill_density :
        ax.fill_between(
            mesures_density/mesures_density.sum(), 
            c14age, 
            mesures, 
            label=label,
            color=color, 
            alpha=.3
        )


    ax.set_xlabel('Probability')
    ax.set_ylabel('$^{14}$C ages in years BP')
    
    return ax




# représentation des résultats de la calibration individuelle :
# densité mesure c14 + courbe de calibration + région HDP date calibrée

def plot_calib_results(
    calibration_results: Dict[str, Any],
    c14age: float = None,
    c14sig: float = None,
    
    plot_BNN: bool = True,
    covariables: bool = None,
    color_BNN: str = 'blue',
    parts_1_and_2: bool = True,
    part_1: bool = False,
    part_2: bool = False,
    
    plot_IntCal20: bool = False,
    color_IntCal20: str = 'green',
    
    figsize = None,
    fontsize_legend: str = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'][2],
    fig = None,
    axs = None,
    add_grid: bool = False,
    
    color_cal_date: str = "cyan",
    eps: float = 10**(-7),
    plot_HPD_bounds: bool = False,
    plot_HPD_threshold: bool = False,
    
    color_c14age: str = "gray",
    support_size: float = 5,
    sample_size: int = 1000,
    plot_density: bool = False,
    fill_density: bool = True
) -> None:
    """
    Plot a full graphical summary of individual radiocarbon calibration results.
    
    The figure combines:

    - the probability density of the radiocarbon measurement,
    - the calibration curve (BNN-based or IntCal20),
    - the posterior density of the calibrated date with its HPD region,
    - a blank panel for layout aesthetics.

    Parameters
    ----------
    calibration_results : Dict[str, Any]
        A dictionary containing all required calibration outputs
        (c14age, c14sig, covariables, posterior density, HPD intervals, etc.).
    c14age : float, optional
        Preserved for backward compatibility; the radiocarbon age (will be overwritten
        by the value inside `calibration_results`).
    c14sig : float, optional
        Preserved for backward compatibility; the radiocarbon age uncertainty (will be 
        overwritten by the value inside `calibration_results`).
    plot_BNN : bool, optional
        If True, plot the BNN calibration curve (unless covariables is None).
    covariables : bool or None, optional
        Preserved for backward compatibility; whether covariates were used for BNN 
        calibration (will be overwritten by the value inside `calibration_results`).  
        If None, IntCal20 is used automatically.
    color_BNN : str, optional
        Color for the BNN calibration curve.
    parts_1_and_2 : bool, optional
        Whether to plot the entire BNN curve (parts 1 and 2).
    part_1 : bool, optional
        If True, only plot part 1 of the BNN calibration curve.
    part_2 : bool, optional
        If True, only plot part 2 of the BNN calibration curve.
    plot_IntCal20 : bool, optional
        If True, plot the IntCal20 curve instead of a BNN model.
    color_IntCal20 : str, optional
        Color for the IntCal20 calibration curve.
    figsize : tuple or None, optional
        Size of the figure in inches. If None, a default size is chosen.
    fontsize_legend : str, optional
        Legend font size.
    fig : matplotlib.figure.Figure or None, optional
        A figure instance to reuse; if None, one is created.
    axs : array-like of Axes or None, optional
        A 2×2 array of axes. If None, axes are created.
    add_grid : bool, optional
        Whether to add grid lines to all plots.
    color_cal_date : str, optional
        Color used for plotting the posterior distribution of the calibrated date.
    eps : float, optional
        Density threshold below which points are considered negligible.  
        A negative value allows to display the posterior distribution of the 
        calibrated date over all the entire range of the calibration curve.
    plot_HPD_bounds : bool, optional
        If True, display vertical HPD interval bounds.
    plot_HPD_threshold : bool, optional
        If True, plot the HPD threshold line.
    color_c14age : str, optional
        Color for the radiocarbon measurement density plot.
    support_size : float, optional
        Half-width of the range (in σ units) used to construct the `c14age` density.
    sample_size : int, optional
        Number of points in the discretized `c14age` density evaluation.
    plot_density : bool, optional
        Whether to draw the `c14age` density curve.
    fill_density : bool, optional
        Whether to fill the area under the `c14age` density curve.

    Returns
    -------
    None
        The function displays the figure directly using `plt.show()`.
    """

    # récupération des paramètres d'entrées et choix automatique entre la courbe BNN et
    #  la courbe IntCal20 en fonction des résultats de calibration
    c14age = calibration_results['c14age']
    c14sig = calibration_results['c14sig']
    covariables = calibration_results['covariables']
    if covariables == None :
        plot_BNN = False
        plot_IntCal20 = True
    else :
        plot_BNN = True
        plot_IntCal20 = False
    
    if figsize == None :
        # choix des unités pour les axes afin de fixer la taille de la figure
        cm = 1/2.54  # centimètres en pouces
        #figsize=(29.7*cm, 29.7*cm)
        figsize=(13*cm, 10.5*cm)
      
    if fig == None and axs == None :
        # subdivision de la grille en 4 parties (2 lignes et 2 colonnes)
        fig, axs = plt.subplots(
            nrows=2, 
            ncols=2, 
            figsize=figsize,
            gridspec_kw={
                'width_ratios': [1, 3],
                'height_ratios': [3, 1]
            }
        )
    
    # plot 1 (axs[0,0]) : distribution de la mesure de laboratoire
    
    add_c14age_density_plot(
        c14age=c14age,
        c14sig=c14sig,
        ax=axs[0,0],
        color=color_c14age,
        support_size=support_size,
        sample_size =sample_size,
        plot_density=plot_density,
        fill_density=fill_density
    )
    
    # plot 2 (axs[0,1]) : Courbe de calibration et incertitude autour de la courbe
    
    ax = axs[0,1]
    if plot_BNN :
        if parts_1_and_2 :
            add_bnn_calibration_curve(
                ax = ax,
                figsize = None,
                color = color_BNN,
                alpha=.3,
                incertitude = True,
                sigma_length = 1,
                Min_x = None, Max_x = None,
                Min_y = None, Max_y = None,
                invert_xaxis = True,
                domaine = ['delta14c', 'c14', 'f14c'][1],
                covariables = covariables,
                credible_interval = False,
                credible_interval_level = 0.95,
                credible_color = None,
                credible_alpha = None
            )
            part_1 = False
            part_2 = False
        elif part_1 :
            add_individual_calibration_curve_part_1(
                ax = ax,
                figsize = None,
                color = color_BNN,
                alpha=.3,
                incertitude = True,
                sigma_length = 1,
                Min_x = None, Max_x = None,
                Min_y = None, Max_y = None,
                invert_xaxis = True,
                domaine = ['delta14c', 'c14', 'f14c'][1],
                covariables = covariables,
                credible_interval = False,
                credible_interval_level = 0.95,
                credible_color = None,
                credible_alpha = None
            )
        else :
            part_2 = True
        
        if part_2 :
            add_individual_calibration_curve_part_2(
                ax = ax,
                figsize = None,
                color = color_BNN,
                alpha=.3,
                incertitude = True,
                sigma_length = 1,
                Min_x = None, Max_x = None,
                Min_y = None, Max_y = None,
                invert_xaxis = True,
                domaine = ['delta14c', 'c14', 'f14c'][1],
                covariables = covariables,
                credible_interval = False,
                credible_interval_level = 0.95,
                credible_color = None,
                credible_alpha = None
            )
    else :
        plot_IntCal20 = True
        
    if plot_IntCal20 :
        add_IntCal20_curve(
            ax = ax,
            figsize = None,
            color = color_IntCal20,
            alpha=.3,
            incertitude = True,
            sigma_length = 1,
            Min_x = None, Max_x = None,
            Min_y = None, Max_y = None,
            invert_xaxis = True,
            domaine = ['delta14c', 'c14', 'f14c'][1]
        )
    
    # plot 3 (axs[1,0]) : Vide
    
    axs[1,0].spines[:].set_visible(False)
    axs[1,0].xaxis.set_ticks([])
    axs[1,0].yaxis.set_ticks([])
    
    # plot 4 (axs[1,1]) : distribution a posteriori sur les dates (date calibrée)
    
    add_cal_date_density_plot_and_HPD_region(
        calibration_results=calibration_results,
        ax=axs[1,1],
        color=color_cal_date,
        eps=eps,
        add_legend=False,
        set_title=False,
        plot_HPD_bounds=plot_HPD_bounds,
        plot_HPD_threshold=plot_HPD_threshold
    )
    
    # paramétrage limites plot 2 
    ax.set_xlim(
        axs[1,1].get_xlim()
    )
    ax.set_ylim(
        axs[0,0].get_ylim()
    )
    ax.set_xlabel('')
    
    # grille
    if add_grid :
        axs[0,0].grid()
        axs[0,1].grid() # == ax.grid() ici car ax = axs[0,1]
        axs[1,1].grid()
    
    # légende et figure
    
    fig.tight_layout()
    #fig.legend(loc="lower left")
    fig.legend(
        loc='lower left', 
        fontsize=fontsize_legend,
        fancybox=True, 
        framealpha=0., 
        bbox_to_anchor=(0., 0.05)
    )
    plt.show()


# fonctions publiques du module
__all__ = [
    # fonctions d'affichage
    "plot_calib_results", 
    "plot_individual_calibration_curve_part_1", 
    "plot_individual_calibration_curve_part_2", 
    "plot_bnn_calibration_curve", 
    "plot_IntCal20_curve"
]


# toutes les fonctions du module
all_functions = [
    "add_IntCal20_curve",
    "plot_IntCal20_curve",
    "find_quantile_beta_opt",
    "compute_credible_intervals_bounds",
    "add_individual_calibration_curve_part_1",
    "plot_individual_calibration_curve_part_1",
    "add_individual_calibration_curve_part_2",
    "plot_individual_calibration_curve_part_2",
    "add_individual_calibration_curve_parts_1_and_2",
    "add_bnn_calibration_curve",
    "plot_bnn_calibration_curve",
    "add_cal_date_density_plot_and_HPD_region",
    "add_c14age_density_plot",
    "plot_calib_results"  
]