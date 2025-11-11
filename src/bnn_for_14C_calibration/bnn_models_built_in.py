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
    get_lib_data_paths,
    load_data, 
    minimax_scaling
)

from typing import (
    Optional,
    Union,
    List
)


# ========================================================================
# génération des chemins vers le cache local ou 
# les données embarquées dans le package
# ========================================================================

paths_results_dict = get_lib_data_paths()

# dossier contenant les données IntCal20
IntCal20_dir = paths_results_dict["IntCal20_dir"]

# dossier contenant les variables exogènes 
covariates_dir = paths_results_dict["covariates_dir"]

# dossiers contenant les prédictions et les poids de modèles BNN
bnn_predictions_dir = paths_results_dict["bnn_predictions_dir"]
bnn_weights_dir = paths_results_dict["bnn_weights_dir"]


# ========================================================================
# conception de l'architecture du réseau des neurones
# ========================================================================


# réseau bayésien avec "sortie déterministe"
def bnn_reg_model(
    batch_size: Optional[int] = None,
    train_size: Optional[int] = None,
    prior = gaussian_prior,
    posterior = independent_gaussian_posterior,
    loss_fn = keras.losses.MeanSquaredError(),
    input_shape: int = 1,
    nb_couches_cachees: int = 1,
    neurones_par_couches: Union[int, List[int], str] = "default",
    activation: Union[str, List[str]] = "relu",
    use_bias: Union[bool, List[bool]] = True,
    dropout: Union[float, List[float], str] = "default",
    last_bias: bool = True,
    optimizer = keras.optimizers.Adam,
    learning_rate: float = 0.001,
    hybrid: bool = False,
    nb_couches_cachees_hybrid: int = 0,
    neurones_par_couches_hybrid: Union[int, List[int], str] = "default",
    activation_hybrid: Union[str, List[str]] = "relu",
    use_bias_hybrid: Union[bool, List[bool]] = True,
    kl_use_exact: bool = False,
    last_hybrid: bool = False,
    activation_of_last_layer: bool = False,
    last_activation: str = "relu",
    metrics: List[str] = ["mean_squared_error", "mean_absolute_error"]
) -> keras.Model:
    """
    Construct a Bayesian neural network (BNN) or hybrid BNN for regression tasks.

    Parameters
    ----------
    batch_size : int or None, optional
        Batch size for training. Used to compute KL weight.
    train_size : int or None, optional
        Total number of training samples. Used to compute KL weight.
    prior : callable
        Function returning a prior distribution over weights.
    posterior : callable
        Function returning a posterior distribution over weights.
    loss_fn : callable
        Loss function to use for model compilation (default MSE).
    input_shape : int
        Number of input features.
    nb_couches_cachees : int
        Number of hidden layers.
    neurones_par_couches : int, list of int, or "default"
        Number of neurons per hidden layer. Can be a single int, list of ints of length `nb_couches_cachees`, or "default".
        If `"default"`, 10 neurons are used for each hidden layer.
    activation : str or list of str
        Activation function(s) for hidden layers.
    use_bias : bool or list of bool
        Whether to use bias in hidden layers.
    dropout : float, list of float, or "default"
        Dropout rate(s) for hidden layers.
        If `"default"`, 0.0 (no dropout) is applied to all hidden layers.
    last_bias : bool
        Whether to use bias in the output layer.
    optimizer : keras.optimizers.Optimizer
        Optimizer to use for model compilation.
    learning_rate : float
        Learning rate for optimizer.
    hybrid : bool
        If True, construct a hybrid model with first layers standard and last layers Bayesian.
    nb_couches_cachees_hybrid : int
        Number of Bayesian hidden layers in hybrid model.
    neurones_par_couches_hybrid : int, list of int, or "default"
        Number of neurons per Bayesian hidden layer in hybrid model.
        If `"default"`, 10 neurons are used for each Bayesian hidden layer.
    activation_hybrid : str or list of str
        Activation function(s) for Bayesian hidden layers.
    use_bias_hybrid : bool or list of bool
        Whether to use bias in Bayesian hidden layers.
    kl_use_exact : bool
        Whether to use exact KL divergence in Bayesian layers.
    last_hybrid : bool
        If True, output layer is Bayesian; otherwise standard.
    activation_of_last_layer : bool
        Whether to apply an activation to the last layer.
    last_activation : str
        Activation function of last layer if `activation_of_last_layer=True`.
    metrics : list of str
        List of metrics for model compilation.

    Returns
    -------
    keras.Model
        Compiled Bayesian or hybrid neural network model ready for training or inference.

    Notes
    -----
    - Bayesian layers are implemented via `tfp.layers.DenseVariational`.
    - KL weight is automatically scaled by `1/nb_batchs`.
    - Hybrid model allows combining standard dense layers and Bayesian layers.
    - Dropout is applied after each hidden layer if rate > 0.
    - Last layer can be standard or Bayesian, with optional activation.
    - `"default"` values:
        - `neurones_par_couches` or `neurones_par_couches_hybrid` → 10 neurons per layer
        - `dropout` → 0.0 (no dropout)
    """
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
    path_to_model_weigths: Union[str, Path] = "last_version",
    covariables: bool = False
) -> keras.Model:
    """
    Rebuild and load the first part of a pre-trained hybrid Bayesian Neural Network (BNN) model 
    from its saved weights. It is typically used as the estimation of the first part of the 
    radiocarbon calibration curve (from 0 to 12309 years BP).

    This function reconstructs the architecture of the first hybrid model and loads 
    its corresponding pre-trained weights from disk.  
    The model is hybrid in structure: all hidden layers are standard deterministic dense layers, 
    but the **output layer** is Bayesian (stochastic).

    Parameters
    ----------
    path_to_model_weigths : str or pathlib.Path, optional
        Path to the file containing the model weights.  
        If set to `"last_version"`, the path is automatically resolved to the latest 
        available version:
        - `"bnn_part_1_with_covariables.weights.h5"` if `covariables=True`
        - `"bnn_part_1_without_covariables.weights.h5"` otherwise.
    covariables : bool, optional
        Whether the model includes exogenous covariates as input features:
        - `True` → `input_shape=3`
        - `False` → `input_shape=1`.

    Returns
    -------
    keras.Model
        The reconstructed hybrid BNN model ready for inference.

    Notes
    -----
    - Model structure:
        - Hidden layers: 5 standard dense layers with sizes `[120, 300, 320, 340, 500]` 
          and ReLU activations.
        - Output layer: 1 Bayesian (stochastic) dense layer.
        - Hidden layer biases: `[False, True, True, False, True]`.
    - Default parameter behavior in the internal function `bnn_reg_model`:
        - `dropout="default"` → dropout rate = 0.0 (no regularization)
        - `neurones_par_couches_hybrid="default"` → 10 neurons per Bayesian layer (if used).
    - This model does **not include any Bayesian hidden layers**, only the last output layer 
      is stochastic (i.e. hybrid configuration with `last_hybrid=True` and `nb_couches_cachees_hybrid=0`).
    """

    # quelques paramètres du modèle à construire (l'architecture du modèle)
    
    nb_couches_cachees = 5
    neurones_par_couches = [120, 300, 320, 340, 500]
    use_bias = [False, True, True, False, True]
    last_bias = True
    kl_use_exact = False

    # création du modèle 
    train_size = None  # on va utiliser un modèle déjà entraîné juste pour faire de la prédiction
    batch_size = None
    if covariables:
        input_shape = 3
    else:
        input_shape = 1

    bnn_model_part_1 = bnn_reg_model(
        train_size=train_size,
        batch_size=batch_size,
        prior=gaussian_prior,
        posterior=independent_gaussian_posterior,
        loss_fn=keras.losses.MeanSquaredError(),
        input_shape=input_shape,
        nb_couches_cachees=nb_couches_cachees,
        neurones_par_couches=neurones_par_couches,
        activation="relu",
        use_bias=use_bias,
        dropout="default",
        last_bias=last_bias,
        optimizer=keras.optimizers.Adam,
        learning_rate=0.001,
        hybrid=True,
        nb_couches_cachees_hybrid=0,  # couches bayésiennes si hybrid vaut True
        neurones_par_couches_hybrid=10,
        kl_use_exact=kl_use_exact,
        last_hybrid=True,
        metrics=["mean_squared_error", "mean_absolute_error"]
    )

    # chargement des poids sauvegardés de ce modèle obtenus lors de l'entraînement
    if path_to_model_weigths == "last_version":
        if covariables:
            model_file_name = "bnn_part_1_with_covariables.weights.h5"
        else:
            model_file_name = "bnn_part_1_without_covariables.weights.h5"
        path_to_model_weigths = bnn_weights_dir / model_file_name

    bnn_model_part_1.load_weights(path_to_model_weigths)
    return bnn_model_part_1



def bnn_load_model_part_2(
    path_to_model_weigths: Union[str, Path] = "last_version",
    covariables: bool = False
) -> keras.Model:
    """
    Rebuild and load the second part of a pre-trained hybrid Bayesian Neural Network (BNN) model 
    from its saved weights. It is typically used as the estimation of the second part of the 
    radiocarbon calibration curve (beyond 12309 years BP).

    This function reconstructs the architecture of the second hybrid model and loads 
    its corresponding pre-trained weights from disk.  
    Unlike the first model, this one includes both a **Bayesian hidden layer** and a 
    **Bayesian output layer**.

    Parameters
    ----------
    path_to_model_weigths : str or pathlib.Path, optional
        Path to the file containing the model weights.  
        If set to `"last_version"`, the path is automatically resolved to the latest 
        available version:
        - `"bnn_part_2_with_covariables.weights.h5"` if `covariables=True`
        - `"bnn_part_2_without_covariables.weights.h5"` otherwise.
    covariables : bool, optional
        Whether the model includes exogenous covariates as input features:
        - `True` → `input_shape=3`
        - `False` → `input_shape=1`.

    Returns
    -------
    keras.Model
        The reconstructed hybrid BNN model ready for inference.

    Notes
    -----
    - Model structure:
        - Hidden layers: 4 standard dense layers `[120, 300, 320, 340]`
          followed by **1 Bayesian hidden layer (500 neurons)**.
        - Output layer: Bayesian (stochastic) dense layer.
        - Hidden layer biases: `[False, True, True, False]`.
        - The 4 standard dense layers'weights are the same weights obtained after the 
            training of the first part of the BNN model (this is a kind of transfer learning).
            Then the training of this model allows to estimate only the Bayesian hidden layer's
            and the last layer's weights.
    - This model is hybrid with both deterministic and stochastic layers.
    - The last two layers are Bayesian: one hidden and one output layer.
    """

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
    train_size = None  # on va utiliser un modèle déjà entraîné juste pour faire de la prédiction
    batch_size = None
    if covariables:
        input_shape = 3
    else:
        input_shape = 1

    bnn_model_part_2 = bnn_reg_model(
        train_size=train_size,
        batch_size=batch_size,
        prior=gaussian_prior,
        posterior=independent_gaussian_posterior,
        loss_fn=keras.losses.MeanSquaredError(),
        input_shape=input_shape,
        nb_couches_cachees=nb_couches_cachees,
        neurones_par_couches=neurones_par_couches,
        activation="relu",
        use_bias=use_bias,
        dropout="default",
        last_bias=last_bias,
        optimizer=keras.optimizers.Adam,
        learning_rate=0.001,
        hybrid=hybrid,
        nb_couches_cachees_hybrid=nb_couches_cachees_hybrid,
        neurones_par_couches_hybrid=neurones_par_couches_hybrid,
        use_bias_hybrid=use_bias_hybrid,
        kl_use_exact=kl_use_exact,
        last_hybrid=True,
        metrics=["mean_squared_error", "mean_absolute_error"]
    )

    # chargement des poids sauvegardés
    if path_to_model_weigths == "last_version":
        if covariables:
            # TODO : vérifier dernière version des points pour la partie 2 du modèle avec covariables
            model_file_name = "bnn_part_2_with_covariables.weights.h5"
        else:
            model_file_name = "bnn_part_2_without_covariables.weights.h5"
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



__all__ = [
    "bnn_reg_model",
    "bnn_load_model_part_1",
    "bnn_load_model_part_2",
    "spline_regressor_built_in",
    "create_and_fit_Be10_curve",
    "create_and_fit_PaleoIntensity_curve",
    "create_features"
]
