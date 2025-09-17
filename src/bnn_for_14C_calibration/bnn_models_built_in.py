# -*- coding: utf-8 -*-


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np

# pour contruire les B-splines
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline

from .bnn_models_built_in_utils import (
    negative_loglikelihood, 
    independent_gaussian_posterior, 
    gaussian_prior
)

from .utils import (
    load_data, 
    minimax_scaling
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
    bnn_weights_dir = dir_path.parents[1] / "models" / "weights"
else :
    # sinon, pour la librairie finale, on utilise les prédictions en cache
    bnn_predictions_dir = cache_dir / "models" / "predictions" / "last_version"
    bnn_weights_dir = cache_dir / "models" / "weights"

# dossier contenant les variables exogènes 
covariates_dir = dir_path / "data" / "exogenous_variables"

# ========================================================================
# conception de l'architecture du réseau des neurones
# ========================================================================


# réseau bayésien avec "sortie déterministe"
def bnn_reg_model(
    batch_size=None, # None : si modèle déjà entrainé et utilisé juste pour faire de l'inférence
    train_size=None, # None : si modèle déjà entrainé et utilisé juste pour faire de l'inférence
    prior = gaussian_prior,
    posterior = independent_gaussian_posterior,
    loss_fn=keras.losses.MeanSquaredError(),
    input_shape=1,
    nb_couches_cachees=1,
    # sinon une liste d'entiers de taille nb_couches_cachees
    neurones_par_couches="default",
    activation="relu",  # sinon une liste de fonctions d'activation de taille nb_couches_cachees
    use_bias=True,  # biais couches cachées : un booléen ou une liste de booléens de taille nb_couches_cachees
    # sinon une liste de taux de dropout (compris entre 0 et 1) de taille nb_couches_cachees
    dropout="default",
    # biais dernière couche (ici avec 1 seul neurones car régression)
    last_bias=True,
    optimizer= keras.optimizers.Adam, # keras.optimizers.RMSprop,  # un optimizer de keras
    learning_rate=0.001,
    hybrid=False,  # mélange bayésien et fréquentiste
    nb_couches_cachees_hybrid=0,  # couches bayésiennes si hybrid vaut True
    # sinon une liste d'entiers de taille nb_couches_cachees
    neurones_par_couches_hybrid="default",
    # sinon une liste de fonctions d'activation de taille nb_couches_cachees
    activation_hybrid="relu",
    # biais couches cachées : un booléen ou une liste de booléens de taille nb_couches_cachees
    use_bias_hybrid=True,
    kl_use_exact=False, # utiliser la divergence de KL analytique ou pas (donc approchée)
    last_hybrid=False, # si True, la couche de sortie est bayésienne ; sinon elle est standard
    activation_of_last_layer=False, # à mettre à trou par exemple si Y est réduite dans [0,1] avec une sigmoide
    last_activation="relu", # activation de la dernière couche au cas où activation_of_last_layer vaut True
    # liste des métriques à utiliser
    metrics=["mean_squared_error", "mean_absolute_error"]
):

    # traitement des paramètres par défaut inchangés

    # nombre de neurones par couche cachée
    if neurones_par_couches == "default":
        default_number = 10
        neurones_par_couches = [default_number]*nb_couches_cachees

    if not(isinstance(neurones_par_couches, list)):
        neurones_par_couches = [int(neurones_par_couches)]*nb_couches_cachees

    # activation
    if not(isinstance(activation, list)):
        activation = [activation]*nb_couches_cachees

    # biais
    if not(isinstance(use_bias, list)):
        use_bias = [use_bias]*nb_couches_cachees

    # dropout
    if dropout == "default":
        default_rate = 0.0
        dropout = [default_rate]*nb_couches_cachees

    if not(isinstance(dropout, list)):
        dropout = [float(dropout)]*nb_couches_cachees

    # nombre de batchs
    if train_size != None and batch_size != None :
        nb_batchs = train_size/batch_size
        if int(nb_batchs) < nb_batchs:  # si nb_batchs n'est pas entier, on l'arrondit à l'entier supérieur
            nb_batchs = int(nb_batchs) + 1
    else :
        nb_batchs = 1

    # initialisation du modèle
    model = keras.Sequential()
    
    if not hybrid :
        # ajout de la première couche cachée
        
        # model.add(keras.Input(shape=(input_shape,))) # autre manière de spécifier le nombre de variables (première couche uniquement)
        model.add(
            tfp.layers.DenseVariational(
                units=neurones_par_couches[0],
                use_bias=use_bias[0],
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1/nb_batchs,
                kl_use_exact=kl_use_exact,
                activation=activation[0],
                input_dim=input_shape
            )
        )
        if dropout[0] > 0 and dropout[0] < 1:
            model.add(layers.Dropout(rate=dropout[0]))
    
        # ajout et paramétrage des autres couches cachées s'il y en a
        if nb_couches_cachees >= 2:
            for i in range(1, nb_couches_cachees):
                model.add(
                    tfp.layers.DenseVariational(
                        units=neurones_par_couches[i],
                        use_bias=use_bias[i],
                        make_prior_fn=prior,
                        make_posterior_fn=posterior,
                        kl_weight=1/nb_batchs,
                        kl_use_exact=kl_use_exact,
                        activation=activation[i]
                    )
                )
                if dropout[i] > 0 and dropout[i] < 1:
                    model.add(layers.Dropout(rate=dropout[i]))

    else : 
        # on créera plutôt un modèle hybrid dont les premières couches cachées seront standards
        # et les dernières sont bayésiennes
        # la dernière couche reste standard ici
        
        # traitement des paramètres hybrid par défaut inchangés

        # nombre de neurones par couche cachée
        if neurones_par_couches_hybrid == "default":
            default_number = 10
            neurones_par_couches_hybrid = [default_number]*nb_couches_cachees_hybrid

        if not(isinstance(neurones_par_couches_hybrid, list)):
            neurones_par_couches_hybrid = [int(neurones_par_couches_hybrid)]*nb_couches_cachees_hybrid

        # activation
        if not(isinstance(activation_hybrid, list)):
            activation_hybrid = [activation_hybrid]*nb_couches_cachees_hybrid

        # biais
        if not(isinstance(use_bias_hybrid, list)):
            use_bias_hybrid = [use_bias_hybrid]*nb_couches_cachees_hybrid
        
        # partie réseaux standards du modèle :
            
        # ajout de la première couche cachée
        model.add(
            layers.Dense(
                units = neurones_par_couches[0],
                activation = activation[0],
                use_bias = use_bias[0],
                input_dim = input_shape
            )
        )
        if dropout[0] > 0 and dropout[0] < 1 :
            model.add(layers.Dropout(rate = dropout[0]))
            
        # ajout et paramétrage des autres couches cachées s'il y en a
        if nb_couches_cachees >= 2 :
            for i in range(1, nb_couches_cachees) :
                model.add(
                    layers.Dense(
                        units = neurones_par_couches[i],
                        activation = activation[i],
                        use_bias = use_bias[i]
                    )
                )
                if dropout[i] > 0 and dropout[i] < 1 :
                    model.add(layers.Dropout(rate = dropout[i]))
                    
        # partie réseaux bayésiens du modèle :
        
        # ajout et paramétrage des couches cachées bayésiennes
        #if nb_couches_cachees_hybrid >= 1:
        for i in range( nb_couches_cachees_hybrid):
            model.add(
                tfp.layers.DenseVariational(
                    units=neurones_par_couches_hybrid[i],
                    use_bias=use_bias_hybrid[i],
                    make_prior_fn=prior,
                    make_posterior_fn=posterior,
                    kl_weight=1/nb_batchs,
                    kl_use_exact=kl_use_exact,
                    activation=activation_hybrid[i]
                )
            )
    
    if not activation_of_last_layer :
        # dernière couche : pas d'activation (= fonction identité par défaut)
        if not last_hybrid : 
            model.add(
                layers.Dense(
                    units=1,
                    use_bias=last_bias
                )
            )
        else :
            model.add(
                tfp.layers.DenseVariational(
                    units=1,
                    use_bias=last_bias,
                    make_prior_fn=prior,
                    make_posterior_fn=posterior,
                    kl_weight=1/nb_batchs,
                    kl_use_exact=kl_use_exact
                )
            )
    else :
        # dernière couche : présence d'activation (pour contrôler la plage de variation des Y ou éviter des gradients très élevés)
        if not last_hybrid : 
            model.add(
                layers.Dense(
                    units=1,
                    use_bias=last_bias,
                    activation=last_activation
                )
            )
        else :
            model.add(
                tfp.layers.DenseVariational(
                    units=1,
                    use_bias=last_bias,
                    make_prior_fn=prior,
                    make_posterior_fn=posterior,
                    kl_weight=1/nb_batchs,
                    kl_use_exact=kl_use_exact,
                    activation=last_activation
                )
            )

    # compilation du modèle
    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss=loss_fn,  # "mean_squared_error", # mse # keras.losses.MeanSquaredError()
        weighted_metrics=[],
        metrics=metrics
    )
    
    return model

# ========================================================================
# chargement des modèles pré-entrainés à partir de leurs poids
# ========================================================================

# pour re-créer et charger un modèle dont les poids sont pré-sauvegardés
def bnn_load_model_part_1(
        path_to_model_weigths = "last_version",
        covariables=False
) :
    
    # quelques paramètres du modèle à construire (l'architecture du modèle)
    
    nb_couches_cachees = 5
    neurones_par_couches = [120, 300, 320, 340, 500]
    use_bias = [False, True, True, False, True]
    last_bias = True
    kl_use_exact = False
    
    
    # création du modèle 
    train_size = None # on va utiliser un modèle déjà entraîné juste pour faire de la prédiction
    batch_size = None # on va utiliser un modèle déjà entraîné juste pour faire de la prédiction
    if covariables :
        input_shape = 3
    else :
        input_shape = 1
    
    bnn_model_part_1 = bnn_reg_model(
        train_size = train_size,
        batch_size = batch_size,
        prior = gaussian_prior,
        posterior = independent_gaussian_posterior,
        loss_fn = keras.losses.MeanSquaredError(),
        input_shape = input_shape,
        nb_couches_cachees = nb_couches_cachees,
        neurones_par_couches = neurones_par_couches,
        activation = "relu",
        use_bias = use_bias,
        dropout = "default",
        last_bias = last_bias,
        optimizer = keras.optimizers.Adam,
        learning_rate = 0.001,
        hybrid = True,
        nb_couches_cachees_hybrid=0,  # couches bayésiennes si hybrid vaut True
        # sinon une liste d'entiers de taille nb_couches_cachees
        neurones_par_couches_hybrid=10,
        kl_use_exact=kl_use_exact,
        last_hybrid = True,
        metrics = ["mean_squared_error", "mean_absolute_error"]
    )

    
    # print("Voici le résumé de l'architecture du modèle construit : \n")
    #bnn_model_part_1_fine_tunned.summary()
    
    # chargement des poids sauvegrdés de ce modèle obtenus lors de l'entraînement
    if path_to_model_weigths == "last_version" :
        if covariables :
            model_file_name ="bnn_part_1_with_covariables.weights.h5"
        else :
            model_file_name ="bnn_part_1_without_covariables.weights.h5"
        path_to_model_weigths = bnn_weights_dir / model_file_name
    bnn_model_part_1.load_weights(path_to_model_weigths)
    
    return bnn_model_part_1




def bnn_load_model_part_2(
        path_to_model_weigths = "last_version",
        covariables=False
) :
    
    # quelques paramètres du modèle à construire (l'architecture du modèle)
    
    nb_couches_cachees = 4
    neurones_par_couches = [120, 300, 320, 340]
    use_bias = [False, True, True, False]
    last_bias = True
    hybrid = True
    kl_use_exact = False
    
    nb_couches_cachees_hybrid = 1
    neurones_par_couches_hybrid = [500]
    use_bias_hybrid = [True]
    
    
    # création du modèle 
    train_size = None # on va utiliser un modèle déjà entraîné juste pour faire de la prédiction
    batch_size = None # on va utiliser un modèle déjà entraîné juste pour faire de la prédiction
    if covariables :
        input_shape = 3
    else :
        input_shape = 1
    
    bnn_model_part_2 = bnn_reg_model(
        train_size = train_size,
        batch_size = batch_size,
        prior = gaussian_prior,
        posterior = independent_gaussian_posterior,
        loss_fn = keras.losses.MeanSquaredError(),
        input_shape = input_shape,
        nb_couches_cachees = nb_couches_cachees,
        neurones_par_couches = neurones_par_couches,
        activation = "relu",
        use_bias = use_bias,
        dropout = "default",
        last_bias = last_bias,
        optimizer = keras.optimizers.Adam,
        learning_rate = 0.001,
        hybrid = hybrid,
        nb_couches_cachees_hybrid=nb_couches_cachees_hybrid,  # couches bayésiennes si hybrid vaut True
        # sinon une liste d'entiers de taille nb_couches_cachees
        neurones_par_couches_hybrid=neurones_par_couches_hybrid,
        use_bias_hybrid=use_bias_hybrid,
        kl_use_exact=kl_use_exact,
        last_hybrid = True,
        metrics = ["mean_squared_error", "mean_absolute_error"]
    )
    
    # print("Voici le résumé de l'architecture du modèle construit : \n")
    #bnn_model_part_1_fine_tunned.summary()
    
    # chargement des poids sauvegrdés de ce modèle obtenus lors de l'entraînement
    if path_to_model_weigths == "last_version" :
        if covariables :
            # TODO : vérifier dernière version des points pour la partie 2 du modèle avec covariables
            model_file_name ="bnn_part_2_with_covariables.weights.h5"
        else :
            model_file_name ="bnn_part_2_without_covariables.weights.h5"
        path_to_model_weigths = bnn_weights_dir / model_file_name
    bnn_model_part_2.load_weights(path_to_model_weigths)
    
    return bnn_model_part_2


# ========================================================================
# conception de la fonction créant l'interpolateur pour les covariables
# (regression par splines)
# ========================================================================

# fonctions à intégrer dans le module model_built_in (ou utils_functions selon préférence)

def spline_regressor_built_in(
    # paramètres de SplineTransfomer
    n_knots=5, # nombre de noeuds >=2
    degree=3, # degré de la spline (cubique par défaut)
    knots='quantile', # méthode des placements de noeuds ('uniforme' par défaut ou 'quantile'), 
                     # sinon un vecteur des noeuds (dans ce cas, n_knots est ignoré)
    extrapolation='constant', # méthode d'extrapolation au délà de la plage d'apprentissage 
                              # {'error', 'constant', 'linear', 'continue', 'periodic'}, default='constant'
    include_bias=True, # inclure un intercept pour chacune des bases de la spline ou pas
    
    # paramètres de Ridge
    alpha=1.0, # pénalisation
    fit_intercept=True # présence ou non de l'intercept dans le modèle de régression
) :
    spline_transformer = SplineTransformer(
        n_knots = n_knots,
        degree = degree,
        knots = knots,
        extrapolation = extrapolation,
        include_bias = include_bias
    )
    
    rigde_regressor = Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept
    )
    
    model = Pipeline([
        ('create_spline_basis', spline_transformer),
        ('make_penalized_linear_regression', rigde_regressor)
    ])
    
    return model


# ========================================================================
# Fonction créant et ajustant la courbe de Be10
#
# il faut convertir les âges GICC05 en âges BP
# âge GICC05 en années BP = âge GICC05 + 50
# ========================================================================

def create_and_fit_Be10_curve(
        Max_age = 55000,
        Min_age = 12310,
        eps = 0.001,
        add_eps = False,
        GICC05_to_BP = True,
        n_knots = 1000, # avec environ 40000 données d'entrainement, ça fait 1 noeud tous 
               # les 40000/1000 = 40 quantiles empiriques
        alpha= 1.0, #1e-3 #1.0
        extrapolation= 'constant', # {'error', 'constant', 'linear', 'continue', 'periodic'}, default='constant'
        file_path = covariates_dir  / "be10.csv"
) :
    Be10_data = load_data(path = file_path)
    
    if GICC05_to_BP :
        Be10_data["age"] = Be10_data["age"] + 50
    
    if add_eps :
        Min_age += eps
    
    Be10_data.loc[:,"calage_scaled"] = np.array(minimax_scaling(Be10_data.loc[:,"age"],Max_age,Min_age))
    
    X_data = np.transpose(np.array(Be10_data.loc[:,"calage_scaled"], ndmin = 2))
    Y_data = np.array(Be10_data.loc[:,"p10Be"])
    
    Be10_curve = spline_regressor_built_in(n_knots=n_knots, extrapolation=extrapolation, alpha=alpha)
    Be10_curve.fit(X_data, Y_data)
    
    return Be10_curve
    

# ========================================================================
# Fonction créant et ajustant la courbe de PaleoIntensite
# ========================================================================

def create_and_fit_PaleoIntensity_curve(
        Max_age = 55000,
        Min_age = 12310,
        eps = 0.001,
        add_eps = False,
        GICC05_to_BP = True,
        n_knots = 77, # avec 393 données d'entrainement, ça fait environ 1 noeud tous 
               # les 393/77 = 5 quantiles empiriques
        alpha=1e-3,
        extrapolation= 'constant', # {'error', 'constant', 'linear', 'continue', 'periodic'}, default='constant'
        file_path = covariates_dir  / "glopis.csv"
) :
    PaleoIntensity_data = load_data(path = file_path)
    
    if GICC05_to_BP :
        PaleoIntensity_data["age"] = PaleoIntensity_data["age"] + 50
    
    if add_eps :
        Min_age += eps
    
    PaleoIntensity_data.loc[:,"calage_scaled"] = np.array(minimax_scaling(PaleoIntensity_data.loc[:,"age"],Max_age,Min_age))
    
    X_data = np.transpose(np.array(PaleoIntensity_data.loc[:,"calage_scaled"], ndmin = 2))
    Y_data = np.array(PaleoIntensity_data.loc[:,"paleo_intensity"])
    
    PaleoIntensity_curve = spline_regressor_built_in(n_knots=n_knots, extrapolation=extrapolation, alpha=alpha)
    PaleoIntensity_curve.fit(X_data, Y_data)
    
    return PaleoIntensity_curve

# ========================================================================
# Fonction générant les covariables
# ========================================================================

def create_features(
        X_train,
        X_val = None,
        X_test = None,
        covariables_list_models = [],
        covariables_max_values_from_training_stage = [],
        covariables_min_values_from_training_stage = [],
        scale_new_variables = True
) :
    n_covariables = len(covariables_list_models)
    
    if len(covariables_max_values_from_training_stage) == 0 and len(covariables_min_values_from_training_stage) == 0 :
        min_and_max_values = False
    else :
        min_and_max_values = True
    
    if scale_new_variables :
        X_train_with_covariables = [X_train]
        for i in range(n_covariables) :
            pred_covariable_i = covariables_list_models[i].predict(X_train).reshape((-1,1))
            
            if min_and_max_values :
                min_covariable_i = covariables_min_values_from_training_stage[i]
                max_covariable_i = covariables_max_values_from_training_stage[i]
            else :
                min_covariable_i = pred_covariable_i.min()
                covariables_min_values_from_training_stage.append(min_covariable_i)
                
                max_covariable_i = pred_covariable_i.max()
                covariables_max_values_from_training_stage.append(max_covariable_i)
                
            pred_covariable_i = minimax_scaling(pred_covariable_i, Max=max_covariable_i, Min=min_covariable_i)
            X_train_with_covariables.append(
                pred_covariable_i
            )
        # X_train_with_covariables  = np.concatenate(X_train_with_covariables, axis=1)
        X_train_with_covariables  = np.hstack(X_train_with_covariables)
    
        if X_val != None :
            X_val_with_covariables = [X_val]
            for i in range(n_covariables) :
                pred_covariable_i = covariables_list_models[i].predict(X_val).reshape((-1,1))
                
                min_covariable_i = covariables_min_values_from_training_stage[i]
                max_covariable_i = covariables_max_values_from_training_stage[i]
                
                pred_covariable_i = minimax_scaling(pred_covariable_i, Max=max_covariable_i, Min=min_covariable_i)
                X_val_with_covariables.append(
                   pred_covariable_i
                )
            X_val_with_covariables  = np.hstack(X_val_with_covariables) 
        else :
            X_val_with_covariables = None
            
        if X_test != None :
            X_test_with_covariables = [X_test]
            for i in range(n_covariables) :
                pred_covariable_i = covariables_list_models[i].predict(X_test).reshape((-1,1))
                
                min_covariable_i = covariables_min_values_from_training_stage[i]
                max_covariable_i = covariables_max_values_from_training_stage[i]
                
                pred_covariable_i = minimax_scaling(pred_covariable_i, Max=max_covariable_i, Min=min_covariable_i)
                X_test_with_covariables.append(
                    pred_covariable_i
                )
            X_test_with_covariables  = np.hstack(X_test_with_covariables)    
        else :
            X_test_with_covariables = None
    else : # voir comment améliorer l'intégration de ce if else sur le minimax scaling des covariables afin de raccourcir ces lignes de
    # dédoublées (mais code efficace ici qu'introduire les "if scale_new_variables :" ) dans les trois boucles "for"
        X_train_with_covariables = [X_train]
        for i in range(n_covariables) :
            pred_covariable_i = covariables_list_models[i].predict(X_train).reshape((-1,1))
            # pred_covariable_i = minimax_scaling(pred_covariable_i, Max=pred_covariable_i.max(), Min=pred_covariable_i.min())
            X_train_with_covariables.append(
                pred_covariable_i
            )
        # X_train_with_covariables  = np.concatenate(X_train_with_covariables, axis=1)
        X_train_with_covariables  = np.hstack(X_train_with_covariables)
    
        if X_val != None :
            X_val_with_covariables = [X_val]
            for i in range(n_covariables) :
                pred_covariable_i = covariables_list_models[i].predict(X_val).reshape((-1,1))
                # pred_covariable_i = minimax_scaling(pred_covariable_i, Max=pred_covariable_i.max(), Min=pred_covariable_i.min())
                X_val_with_covariables.append(
                   pred_covariable_i
                )
            X_val_with_covariables  = np.hstack(X_val_with_covariables) 
        else :
            X_val_with_covariables = None
            
        if X_test != None :
            X_test_with_covariables = [X_test]
            for i in range(n_covariables) :
                pred_covariable_i = covariables_list_models[i].predict(X_test).reshape((-1,1))
                # pred_covariable_i = minimax_scaling(pred_covariable_i, Max=pred_covariable_i.max(), Min=pred_covariable_i.min())
                X_test_with_covariables.append(
                    pred_covariable_i
                )
            X_test_with_covariables  = np.hstack(X_test_with_covariables)    
        else :
            X_test_with_covariables = None
    
    return X_train_with_covariables, X_val_with_covariables, X_test_with_covariables, covariables_max_values_from_training_stage, covariables_min_values_from_training_stage


