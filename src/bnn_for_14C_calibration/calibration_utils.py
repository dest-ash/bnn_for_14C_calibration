# -*- coding: utf-8 -*-


import numpy as np


# =============================================================================
# approximation de la densité a posteriori des dates calibrées
# =============================================================================

# version calibration individuelle d'une seule date

def mono_cal_date_approx_density(mesure, lab_error, bnn_model, nb_curves=100, prior_density="default", batch_size = None):
    
    # traitement de la densité à piori :
    if prior_density == "default" :
        support_lower_bound = 0. # à remplacer par min_Xtrain ou min_Xtrain_val ou min_Xtrain_val_test plus tard suivant le cas ou date minimale gobale possible pour la calibration
        support_upper_bound = 1. # à remplacer par max_Xtrain ou max_Xtrain_val ou max_Xtrain_val_test plus tard suivant le cas ou date maximale gobale possible pour la calibration
        prior_density = lambda d : np.float64((support_lower_bound <= d) * (d <= support_upper_bound))/(support_upper_bound - support_lower_bound)
    else :
        raise NotImplementedError("La densité a piori fournie sur les dates n'est pas encore supportée")
        
    # predictions avec le modèle
    predicted = lambda d : bnn_make_predictions_(bnn_model = bnn_model, X_test = d.reshape((-1,1)), iterations = nb_curves, batch_size = batch_size)
    
    # densité approchée (connue à une constante près)
    density = lambda d : prior_density(d) * np.exp(-(mesure - predicted(d))**2/(2*lab_error**2)).mean(axis=1, dtype=np.float64) / (lab_error * np.sqrt(2*np.pi))
    
    return density



# rédéfinition de la fonction mono_cal_date_approx_density pour les points milieux afin de tenir
# compte du cas où les prédictions aux points milieux sont déjà pré-calculés pour améliorer
# la rapidité des calculs en cas de plusieurs calibrations avec la même subdivision de l'intervalle [a,b]

def _mono_cal_date_approx_density_on_middle_points_(
    mesure, lab_error, bnn_model=None, middle_points_predictions=None, 
    nb_curves=100, prior_density="default", batch_size = None,
    mesure_likelihood = [
        "gaussian_mixture", "curve_gaussian_approximation", "IntCal20", "exact_gaussian_density"
    ][0], # par défaut : "gaussian_mixture"
    true_regression_function = None # à définir obligatoirement si mesure_likelihood = "exact_gaussian_density"
):
    # vérification paramètres
    if bnn_model == None and type(middle_points_predictions) == 'NoneType' :
        raise ValueError("au moins l'un des arguments 'bnn_model' ou 'middle_points_predictions' doit être fourni (!= None)")
    
    if type(middle_points_predictions) != 'NoneType' and type(middle_points_predictions) != np.ndarray :
        raise ValueError("'middle_points_predictions' doit être de type 'numpy.ndarray' quand il est fourni")
    
    # traitement de la densité à piori :
    if prior_density == "default" :
        support_lower_bound = 0. # à remplacer par min_Xtrain ou min_Xtrain_val ou min_Xtrain_val_test plus tard suivant le cas ou date minimale gobale possible pour la calibration
        support_upper_bound = 1. # à remplacer par max_Xtrain ou max_Xtrain_val ou max_Xtrain_val_test plus tard suivant le cas ou date maximale gobale possible pour la calibration
        prior_density = lambda d : np.float64((support_lower_bound <= d) * (d <= support_upper_bound))/(support_upper_bound - support_lower_bound)
    else :
        support_lower_bound = prior_density[0]
        support_upper_bound = prior_density[1]
        #raise NotImplementedError("La densité à piori fournie sur les dates n'est pas encore supportée")
        
    # predictions avec le modèle
    if type(middle_points_predictions) != 'NoneType' :
        predicted = lambda d : middle_points_predictions
    else : 
        predicted = lambda d : bnn_make_predictions_(bnn_model = bnn_model, X_test = d.reshape((-1,1)), iterations = nb_curves, batch_size = batch_size)
    
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
                f"'true_regression_function' doit être fourni lorsque \n 'mesure_likelihood' vaut '{mesure_likelihood_possible_values[2]}'"
            )
        density = lambda d : prior_density(d) * np.exp(
            -(mesure - true_regression_function(d))**2 / (2*lab_error**2)
        ) / (lab_error * np.sqrt(2*np.pi))
    
    else :
        raise ValueError(
            f"'mesure_likelihood' doit être un élément de la liste \n{mesure_likelihood_possible_values}"
        )
        
    return density




# version calibration simultanée de plusieurs dates

def multi_cal_date_approx_density(mesures, lab_errors, bnn_model, nb_curves=100, prior_density="default", batch_size = None):
    
    dim_dates = mesures.shape[0] # = len(mesures)
    # traitement de la densité à piori :
    if prior_density == "default" :
        support_lower_bound = np.array([0.] * dim_dates) # à remplacer par min_Xtrain ou min_Xtrain_val ou min_Xtrain_val_test plus tard suivant le cas ou date minimale gobale possible pour la calibration
        support_upper_bound = np.array([1.] * dim_dates) # à remplacer par max_Xtrain ou max_Xtrain_val ou max_Xtrain_val_test plus tard suivant le cas ou date maximale gobale possible pour la calibration
        prior_density = lambda d : np.float64((support_lower_bound <= d) * (d <= support_upper_bound)).prod(axis=1)/(support_upper_bound - support_lower_bound).prod()
    else :
        raise NotImplementedError("La densité a piori fournie sur les dates n'est pas encore supportée")
        
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
# (re-définition de la fonction pour tenir compte du cas où on a une 
# relation d'ordre sur les dates)
# (on tient aussi compte du cas où les prédictions aux dates 
# sont déjà disponibles dans le domaine F14C)
# =============================================================================

def _multi_cal_date_approx_density_(
    mesures, lab_errors, 
    
    # les valeurs max et min permettant de transformer les dates de [0,1] vers l'intervalle [Min, Max]
    # utilisé lors de l'entraînement du modèle bayésien
    Max = None,
    Min = None,
    
    bnn_model=None, nb_curves=100, prior_density="default", 
    ordered = False,
    batch_size = None
):
    
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
        raise NotImplementedError("La densité à piori fournie sur les dates n'est pas encore supportée")
        
    # densité approchée (connue à une constante près)
    mesures_broadcasted = mesures.repeat(nb_curves).reshape((-1,nb_curves))
    lab_errors_broadcasted = lab_errors.repeat(nb_curves).reshape(-1,nb_curves)
    
    if bnn_model != None :
        if Max == None or Min == None :
            raise ValueError("Les arguments Max et Min doivent être fournis lorsque bnn_model est fourni")
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

def mono_cal_date_approx_density_sample(density=None, nb_intervals=1000, support_bounds=(0,1), subdivision_components=None, sample_size=1):
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

def mono_cal_date_approx_cumulative_fct(density=None, nb_intervals=1000, support_bounds=(0,1), subdivision_components=None):
    
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



