# -*- coding: utf-8 -*-
import numpy as np

from matplotlib import pyplot as plt
import pandas as pd


from .bnn_models_built_in_utils import (
    #bnn_make_predictions_, 
    bnn_load_predictions_
)

from .utils import (
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

# gestion des chemins relatifs pour différents téléchargements et sauvegardes
from pathlib import Path

# dossier contenant les scripts et données embarquées dans la librairie
dir_path = Path(__file__).resolve().parent 
# dir_path = "chemin_absolu_vers_src/bnn_for_14C_calibration_c14"

# dossier contenant les données IntCal20
IntCal20_dir = dir_path / "data" / "IntCal20"

# dossier contenant les prédictions pré-sauvegardées de différents réseaux de neurones bayésiens
cache_dir = Path.home() / ".bnn_for_14C_calibration"
cache_dir = Path(cache_dir).resolve()
if not cache_dir.exists():
    # pour tester en local depuis le repo git avant construction de la librairie, utiliser le chemin suivant : 
    bnn_predictions_dir = dir_path.parents[1] / "models" / "predictions" / "last_version"
else :
    # sinon, pour la librairie finale, on utilise les prédictions en cache
    bnn_predictions_dir = cache_dir / "models" / "predictions" / "last_version"

# dossier contenant les variables exogènes 
#covariates_dir = dir_path / "data" / "exogenous_variables"



# =============================================================================
# Représentations graphiques 
# =============================================================================



# =============================================================================
# Courbes de calibration
# =============================================================================


# courbe IntCal20
def add_IntCal20_curve(
    ax = None,
    figsize = None,
    color = "green",
    alpha=.4,
    incertitude = True,
    sigma_length = 1,
    Min_x = None, Max_x = None,
    Min_y = None, Max_y = None,
    invert_xaxis = True,
    domaine = ['delta14c', 'c14', 'f14c'][0]
):
    
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
    xlabel = "âges calibrés en années BP",
    ylabel = "âges C14 en années BP",
    add_grid = True,
    fontsize_legend = 'small',
    reset_margins = False,

    ax = None,
    figsize = None,
    color = "green",
    alpha=.4,
    incertitude = True,
    sigma_length = 1,
    Min_x = None, Max_x = None,
    Min_y = None, Max_y = None,
    invert_xaxis = True,
    domaine = ['delta14c', 'c14', 'f14c'][0]
):
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
def find_quantile_beta_opt(alpha, predictions, method='median_unbiased'):
    interval_length = lambda beta : np.quantile(predictions, 1 - alpha + beta, method=method) - np.quantile(predictions, beta, method=method)
    beta_opt = minimize(fun = interval_length , x0 = np.array([alpha/2]), method = 'Nelder-Mead', bounds = [(0.,alpha)])
    return beta_opt.x[0]

# fonction pour calculer les bornes des intervalles de crédibilité pour les prédictions
# = fonction pour calculer l'intervalle de crédibilité autour de la courbe
def compute_credible_intervals_bounds(alpha, bnn_predictions, method='median_unbiased') :
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
    ax = None,
    figsize = None,
    color = 'cyan',
    alpha=.4,
    incertitude = True,
    sigma_length = 1,
    Min_x = None, Max_x = None,
    Min_y = None, Max_y = None,
    invert_xaxis = True,
    domaine = ['delta14c', 'c14', 'f14c'][0],
    middle_points_predictions_part_2_filepath = "last_version",
    covariables = False,
    label_prefix = '',
    label_suffix = '',
    credible_interval = False,
    credible_interval_level = 0.95,
    credible_color = None,
    credible_alpha = None
) :
    
    # chargement des prédictions pré-sauvergardées
    
    if middle_points_predictions_part_2_filepath == "last_version" :
        if covariables :
            filename = "bnn_part_1_with_covariables_middle_points_predictions.csv"
        else :
            filename = "bnn_part_1_without_covariables_middle_points_predictions.csv"
        filepath = bnn_predictions_dir /  filename
        middle_points_predictions_part_2_filepath = filepath
    
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
    xlabel = "âges calibrés en années BP",
    ylabel = "âges C14 en années BP",
    add_grid = True,
    fontsize_legend = 'small',
    ax = None,
    figsize = None,
    color = 'cyan',
    alpha=.4,
    incertitude = True,
    sigma_length = 1,
    Min_x = None, Max_x = None,
    Min_y = None, Max_y = None,
    invert_xaxis = True,
    domaine = ['delta14c', 'c14'][0],
    middle_points_predictions_part_2_filepath = "last_version",
    covariables = False,
    credible_interval = False,
    credible_interval_level = 0.95,
    credible_color = None,
    credible_alpha = None
):
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
            middle_points_predictions_part_2_filepath = middle_points_predictions_part_2_filepath,
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
    ax = None,
    figsize = None,
    color = 'cyan',
    alpha=.4,
    incertitude = True,
    sigma_length = 1,
    Min_x = None, Max_x = None,
    Min_y = None, Max_y = None,
    invert_xaxis = True,
    domaine = ['delta14c', 'c14'][0],
    middle_points_predictions_part_2_filepath = "last_version",
    covariables = False,
    label_prefix = '',
    label_suffix = '',
    credible_interval = False,
    credible_interval_level = 0.95,
    credible_color = None,
    credible_alpha = None
) :
    
    # chargement des prédictions pré-sauvergardées
    
    if middle_points_predictions_part_2_filepath == "last_version" :
        if covariables :
            filename = "bnn_part_2_with_covariables_middle_points_predictions.csv"
        else :
            filename = "bnn_part_2_without_covariables_middle_points_predictions.csv"
        filepath = bnn_predictions_dir /  filename
        middle_points_predictions_part_2_filepath = filepath
    
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
    xlabel = "âges calibrés en années BP",
    ylabel = "âges C14 en années BP",
    add_grid = True,
    fontsize_legend = 'small',
    ax = None,
    figsize = None,
    color = 'cyan',
    alpha=.4,
    incertitude = True,
    sigma_length = 1,
    Min_x = None, Max_x = None,
    Min_y = None, Max_y = None,
    invert_xaxis = True,
    domaine = ['delta14c', 'c14'][0],
    middle_points_predictions_part_2_filepath = "last_version",
    covariables = False,
    credible_interval = False,
    credible_interval_level = 0.95,
    credible_color = None,
    credible_alpha = None
):
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
            middle_points_predictions_part_2_filepath = middle_points_predictions_part_2_filepath,
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
    ax = None,
    figsize = None,
    color = 'cyan',
    alpha=.4,
    incertitude = True,
    sigma_length = 1,
    Min_x = None, Max_x = None,
    Min_y = None, Max_y = None,
    invert_xaxis = True,
    domaine = ['delta14c', 'c14'][0],
    middle_points_predictions_filepath = "last_version",
    covariables = False,
    credible_interval = False,
    credible_interval_level = 0.95,
    credible_color = None,
    credible_alpha = None
) :

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
            middle_points_predictions_part_2_filepath = middle_points_predictions_filepath,
            covariables = covariables,
            label_prefix = '', # la légende apparaît
            label_suffix = label_suffix, # gestion de la présence ou non des covariables dans la légende
            credible_interval = credible_interval,
            credible_interval_level = credible_interval_level,
            credible_color = credible_color,
            credible_alpha = credible_alpha
    )

    if invert_xaxis :
        # on inverse plus l'axe une deuxième fois
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
            middle_points_predictions_part_2_filepath = middle_points_predictions_filepath,
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
    ax = None,
    figsize = None,
    color = 'cyan',
    alpha=.4,
    incertitude = True,
    sigma_length = 1,
    Min_x = None, Max_x = None,
    Min_y = None, Max_y = None,
    invert_xaxis = True,
    domaine = ['delta14c', 'c14'][0],
    middle_points_predictions_filepath = "last_version",
    covariables = False,
    label_prefix = '',
    label_suffix = '',
    credible_interval = False,
    credible_interval_level = 0.95,
    credible_color = None,
    credible_alpha = None
) :

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
    xlabel = "calendar dates in years BP",
    ylabel = "$^{14}$C ages in years BP",
    add_grid = True,
    fontsize_legend = 'small',
    reset_margins = False,

    ax = None,
    figsize = None,
    color = 'cyan',
    alpha=.4,
    incertitude = True,
    sigma_length = 1,
    Min_x = None, Max_x = None,
    Min_y = None, Max_y = None,
    invert_xaxis = True,
    domaine = ['delta14c', 'c14', 'f14c'][0],
    middle_points_predictions_filepath = "last_version",
    covariables = False,
    credible_interval = False,
    credible_interval_level = 0.95,
    credible_color = None,
    credible_alpha = None
):
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
            middle_points_predictions_filepath = middle_points_predictions_filepath,
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
    calibration_results,
    ax = None,
    color = "cyan", #"blue", #"green"
    eps = 10**(-7), # en dessous de ce seuil, la densité est considérée nulle dans le graphique
    set_title = True,
    add_legend = True,
    plot_HPD_bounds = False,
    plot_HPD_threshold = False
) :
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
    c14age,
    c14sig,
    ax = None,
    color = "gray", #"blue", #"green"
    #eps = 10**(-7), # en dessous de ce seuil, la densité est considérée nulle dans le graphique
    support_size = 5, # demi-longueur de l'intervalle sur lequel tracer la densité de l'âge c14
    sample_size = 1000,
    plot_density = False,
    fill_density = True
) :
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




# représentation des résultats de la calibration :
# densité mesure c14 + courbe de calibration + région HDP date calibrée

def plot_calib_results(
    calibration_results=None,
    c14age=None,
    c14sig=None,
    
    # courbe à afficher
    plot_BNN = True,
    covariables = None,
    color_BNN = 'blue',
    parts_1_and_2 = True,
    part_1 = False,
    part_2 = False,
    
    
    plot_IntCal20 = False,
    color_IntCal20 = 'green',
    
    # taille et paramètres de la figure
    figsize = None,
    fontsize_legend = ['xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'][2],
    fig = None,
    axs = None,
    add_grid = False,
    
    # autres paramètres pour la date calibrée
    color_cal_date = "cyan", #"blue", #"green"
    eps = 10**(-7), # en dessous de ce seuil, la densité est considérée nulle dans le graphique
    plot_HPD_bounds = False,
    plot_HPD_threshold = False,
    
    # autres paramètres pour la mesure c14
    color_c14age = "gray", #"blue", #"green"
    support_size = 5, # demi-longueur de l'intervalle sur lequel tracer la densité de l'âge c14
    sample_size = 1000,
    plot_density = False,
    fill_density = True
):

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
                middle_points_predictions_filepath = "last_version",
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
                middle_points_predictions_part_2_filepath = "last_version",
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
                middle_points_predictions_part_2_filepath = "last_version",
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

