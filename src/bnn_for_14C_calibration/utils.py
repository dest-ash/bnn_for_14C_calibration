# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from .manage_cache import(
    CACHE_DIR,
    download_cache_lib_data
)

from typing import (
    # Optional, 
    Union, 
    # List, 
    Dict, 
    Tuple, 
    Any
)

NumberOrArray = Union[float, np.ndarray]

# --------------------------
# Paths and data loaders
# --------------------------

def get_lib_data_paths() -> Dict[str, Path]:
    """
    Generate paths to embedded package data or local cache data.

    Returns
    -------
    Dict[str, Path]
        Dictionary containing paths to:
        - IntCal20 data
        - Exogenous variables
        - Bayesian neural network predictions and weights (from local cache if exists)

    Notes
    -----
    - If the cache does not exist, it will automatically be created using `download_cache_lib_data`.
    """

    # dossier contenant les scripts et données embarquées dans la librairie
    # c.a.d dir_path = "chemin_absolu_vers_src/bnn_for_14C_calibration_c14"
    dir_path = Path(__file__).resolve().parent 
    
    # ========================================================================
    # embedded package data in src/bnn_for_14C_calibration/data : 
    # ========================================================================
    
    ## TO DO : 
    ## manage embedded package data in src/bnn_for_14C_calibration/data with 
    ## import importlib.resources as pkg_resources
    ## and see how to modify paths generated here and their use in the package

    # dossier contenant les données IntCal20
    IntCal20_dir = dir_path / "data" / "IntCal20"

    # dossier contenant les variables exogènes 
    covariates_dir = dir_path / "data" / "exogenous_variables"

    # ========================================================================
    # package data (to be) stored in local cache 
    # ========================================================================

    # dossier contenant les prédictions pré-sauvegardées de différents réseaux de neurones bayésiens
    if not (CACHE_DIR.exists() and CACHE_DIR.is_dir()):
        # pour tester en local depuis le repo git avant construction de la librairie, utiliser le chemin suivant : 
        bnn_predictions_dir = dir_path.parents[1] / "models" / "predictions" / "last_version"
        bnn_weights_dir = dir_path.parents[1] / "models" / "weights"

        # si l'un des chemins locaux spécifié ci-dessus n'existe pas, 
        # on crée le cache local et on re-définit les chemins en utilisant le cache créé
        if not (
            (
                bnn_predictions_dir.exists() and bnn_predictions_dir.is_dir()
            ) or (
                bnn_weights_dir.exists() and bnn_weights_dir.is_dir()
            )
        ):
            download_cache_lib_data()
            bnn_predictions_dir = CACHE_DIR / "models" / "predictions" / "last_version"
            bnn_weights_dir = CACHE_DIR / "models" / "weights"
    else :
        # sinon, pour la librairie finale, on utilise les prédictions en cache
        bnn_predictions_dir = CACHE_DIR / "models" / "predictions" / "last_version"
        bnn_weights_dir = CACHE_DIR / "models" / "weights"

    paths_results_dict = {
        "IntCal20_dir" : IntCal20_dir,
        "covariates_dir" : covariates_dir,
        "bnn_predictions_dir" : bnn_predictions_dir,
        "bnn_weights_dir" : bnn_weights_dir
    }
    
    return paths_results_dict




def read_params_from_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read parameters from a text file and convert them to proper Python types.

    Parameters
    ----------
    file_path : str or Path
        Path to the file containing parameters in 'key : value' format.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping parameter names to their values, converted to
        int, float, bool, or str depending on content.

    Notes
    -----
    - Boolean values are recognized as "True" or "False".
    - Float values are recognized for keys ['alpha', 'beta', 'min_delta'].
    - Other values are attempted to convert to int, otherwise stored as str.
    """
    keys: Dict[str, Any] = {}
    with open(file_path, 'r') as file :
        for line in file :
            key,value = line.strip().split(sep=" : ")
            
            # booléens
            if value == "True" :
                value = True
            elif value == "False" :
                value = False
            
            # floattants
            elif key in ['alpha', 'beta', 'min_delta'] :
                value = float(value)
            
            # entiers ou chaînes de caractères
            else :
                try :
                    # entiers
                    value = int(value)
                except ValueError as e :
                    # chaînes de caractère (ou erreurs non prévues)
                    #if __name__ == "__main__" :
                    print(f"""
                            Ignored error : {e} \n
                            If the parameter {key} is supposed to be a string, this error is normal and ignored. \n
                            Otherwise, this may be a non-handled case and the parameter will be stored as a string.
                    """)
                    pass
            
            keys[key] = value
    return keys


def load_data(
    path: Union[str, Path], 
    sep: str = ";"
) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    sep : str, optional
        Separator used in the CSV file (default ';').

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    dataset = pd.read_csv(path, sep =sep)
    return dataset


# --------------------------
# Min-max scaling
# --------------------------

def minimax_scaling(
    x: NumberOrArray, 
    Max: float, 
    Min: float
) -> NumberOrArray:
    """
    Apply manual min-max scaling to a single value or numpy array.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s).
    Max : float
        Maximum of the original range.
    Min : float
        Minimum of the original range.

    Returns
    -------
    float or np.ndarray
        Scaled value(s) (in [0,1] if Min <= x <= Max).

    Notes
    -----
    - If x is a numpy array, the operation is applied element-wise.
    """
    return (x-Min)/(Max-Min)


def minimax_scaling_reciproque(x: NumberOrArray, Max: float, Min: float) -> NumberOrArray:
    """
    Inverse min-max scaling for a value or numpy array.

    Parameters
    ----------
    x : float or np.ndarray
        Scaled value(s).
    Max : float
        Maximum of the original range.
    Min : float
        Minimum of the original range.

    Returns
    -------
    float or np.ndarray
        Original value(s) before scaling (in [Min,Max] if 0 <= x <= 1).

    Notes
    -----
    - If x is a numpy array, the operation is applied element-wise.
    """
    return (Max-Min)*x + Min


# -------------------------------------------
# Conversion functions c14 <-> d14c <-> f14c
# -------------------------------------------

# ==================================================================================================
# Relations entre les différents domaines du C14

# # c14 = domaine des âges C14
# # f14c = domaine d'acquisition des mesures C14 (i.e. le rapport isotopique C14/C12)
# # d14c = delta c14
# # teta = date (age calendaire/calibré : calage)

# # à chaque domaine est associée l'incertitude de laboratoire : les formules de sa conversion 
# # d'un domaine à l'autre découlent de la delta-méthode (par exemple)
# ==================================================================================================

# domaine d14c vers domaine f14c

def d14c_to_f14c(
    d14c: NumberOrArray, 
    teta: NumberOrArray
) -> NumberOrArray:
    """
    Convert d14c domain to f14c domain.

    Parameters
    ----------
    d14c : float or np.ndarray
        d14c value(s).
    teta : float or np.ndarray
        Calendar age(s), must match the shape of d14c.

    Returns
    -------
    float or np.ndarray
        Corresponding f14c value(s).

    Notes
    -----
    - Supports element-wise operations.
    - Uses transformation formula: 
        f14c = (1 + d14c/1000) * exp(-teta/8267)
    """
    f14c = (1/1000*d14c + 1)*np.exp(-teta/8267)
    return f14c

def d14csig_to_f14csig(
    d14csig: NumberOrArray, 
    teta: NumberOrArray
) -> NumberOrArray:
    """
    Convert d14c uncertainty to f14c uncertainty.

    Parameters
    ----------
    d14csig : float or np.ndarray
        Uncertainty in d14c.
    teta : float or np.ndarray
        Calendar age(s), same shape as d14csig.

    Returns
    -------
    float or np.ndarray
        Corresponding uncertainty in f14c.

    Notes
    -----
    - Supports element-wise operations.
    - Transformation formula can be derived directly or by using delta-method: 
        f14csig = d14csig * exp(-teta/8267)/1000
    """
    f14csig = d14csig*np.exp(-teta/8267)/1000
    return f14csig


# domaine f14c vers domaine d14c

def f14c_to_d14c(
    f14c: NumberOrArray, 
    teta: NumberOrArray
) -> NumberOrArray:
    """
    Convert f14c to d14c.

    Parameters
    ----------
    f14c : float or np.ndarray
        f14c value(s).
    teta : float or np.ndarray
        Calendar age(s), same shape as f14c.

    Returns
    -------
    float or np.ndarray
        Corresponding d14c value(s).

    Notes
    -----
    - Supports element-wise operations.
    - As the inverse function of `d14c_to_f14c`, it uses its formula's inverse for transformation:
        d14c = 1000 * (-1 + f14c * exp(teta/8267))
    """
    d14c = 1000*(-1 + f14c*np.exp(teta/8267))
    return d14c

def f14csig_to_d14csig(
    f14csig: NumberOrArray, 
    teta: NumberOrArray
) -> NumberOrArray:
    """
    Convert f14c uncertainty to d14c uncertainty.

    Parameters
    ----------
    f14csig : float or np.ndarray
        Uncertainty in f14c.
    teta : float or np.ndarray
        Calendar age(s), same shape as f14csig.

    Returns
    -------
    float or np.ndarray
        Corresponding uncertainty in d14c.

    Notes
    -----
    - Supports element-wise operations.
    - As the inverse function of `d14csig_to_f14csig`, it uses its formula's inverse for transformation:
        d14csig = 1000 * f14csig * exp(teta/8267)
    """
    d14csig = 1000*f14csig*np.exp(teta/8267)
    return d14csig


# domaine f14c vers domaine c14

def f14c_to_c14(f14c: NumberOrArray) -> NumberOrArray:
    """
    Convert f14c domain to radiocarbon age ($^{14}$C).

    Parameters
    ----------
    f14c : float or np.ndarray
        f14c value(s).

    Returns
    -------
    float or np.ndarray
        Radiocarbon age(s) $^{14}$C.

    Notes
    -----
    - Supports element-wise operations.
    - Uses transformation formula: 
        c14 = -8033 * log(f14c)
    """
    c14 = -8033*np.log(f14c)
    return c14

def f14csig_to_c14sig(
    f14c: NumberOrArray, 
    f14csig: NumberOrArray
) -> NumberOrArray:
    """
    Convert f14c uncertainty to $^{14}$C uncertainty using delta-method.

    Parameters
    ----------
    f14c : float or np.ndarray
        f14c value(s).
    f14csig : float or np.ndarray
        Uncertainty in f14c, same shape as f14c.

    Returns
    -------
    float or np.ndarray
        Corresponding uncertainty in $^{14}$C.

    Notes
    -----
    - Supports element-wise operations.
    - Transformation formula cannot be derived directly but by using delta-method: 
        c14sig = f14csig * 8033/f14c
    """
    c14_sig = f14csig*8033/f14c # f14c > 0
    return c14_sig


# domaine c14 vers domaine f14c

def c14_to_f14c(c14: NumberOrArray) -> NumberOrArray:
    """
    Convert radiocarbon age ($^{14}$C) to f14c domain.

    Parameters
    ----------
    c14 : float or np.ndarray
        Radiocarbon age(s).

    Returns
    -------
    float or np.ndarray
        Corresponding f14c value(s).

    Notes
    -----
    - Supports element-wise operations.
    - As the inverse function of `f14c_to_c14`, it uses its formula's inverse for transformation:
        f14c = exp(-c14/8033)
    """
    f14c = np.exp(-c14/8033)
    return f14c

def c14sig_to_f14csig(
    c14: NumberOrArray, 
    c14sig: NumberOrArray
) -> NumberOrArray:
    """
    Convert c14 uncertainty to f14c uncertainty using delta-method.

    Parameters
    ----------
    c14 : float or np.ndarray
        Radiocarbon age(s).
    c14sig : float or np.ndarray
        Uncertainty in c14, same shape as c14.

    Returns
    -------
    float or np.ndarray
        Corresponding uncertainty in f14c.

    Notes
    -----
    - Supports element-wise operations.
    - As the inverse function of `f14csig_to_c14sig`, it uses its formula's inverse for transformation:
        f14csig = c14sig * f14c/8033, 
        where f14c is computed using the function `c14_to_f14c`.
    """
    f14c = c14_to_f14c(c14 = c14)
    f14c_sig = c14sig*f14c/8033 # f14c > 0
    return f14c_sig


# domaine d14c vers domaine c14

def d14c_to_c14(
    d14c: NumberOrArray, 
    teta: NumberOrArray
) -> NumberOrArray:
    """
    Convert d14c domain to radiocarbon age ($^{14}$C).

    Parameters
    ----------
    d14c : float or np.ndarray
        d14c value(s).
    teta : float or np.ndarray
        Calendar age(s), same shape as d14c.

    Returns
    -------
    float or np.ndarray
        Corresponding c14 value(s).

    Notes
    -----
    - Supports element-wise operations.
    - This function is computed as the composition of `f14c_to_c14` and `d14c_to_f14c`:
        c14 = f14c_to_c14(d14c_to_f14c(d14c,teta)).
    """
    return f14c_to_c14(d14c_to_f14c(d14c,teta))

def d14csig_to_c14sig(
    d14c: NumberOrArray, 
    d14csig: NumberOrArray, 
    teta: NumberOrArray
) -> NumberOrArray:
    """
    Convert d14c uncertainty to $^{14}$C uncertainty.

    Parameters
    ----------
    d14c : float or np.ndarray
        d14c value(s).
    d14csig : float or np.ndarray
        Uncertainty in d14c.
    teta : float or np.ndarray
        Calendar age(s), same shape as d14c.

    Returns
    -------
    float or np.ndarray
        Corresponding uncertainty in $^{14}$C.

    Notes
    -----
    - Supports element-wise operations.
    - This function is computed as the composition of `f14csig_to_c14sig`, 
        `d14c_to_f14c` and `d14csig_to_f14csig`:
        c14sig = f14csig_to_c14sig(
            d14c_to_f14c(d14c,teta),
            d14csig_to_f14csig(d14csig,teta)
        ).
    """
    return f14csig_to_c14sig(
        d14c_to_f14c(d14c,teta),
        d14csig_to_f14csig(d14csig,teta)
    )


# domaine c14 vers domaine d14c

def c14_to_d14c(c14: NumberOrArray, teta: NumberOrArray) -> NumberOrArray:
    """
    Convert radiocarbon age (c14) to d14c domain.

    Parameters
    ----------
    c14 : float or np.ndarray
        Radiocarbon age(s).
    teta : float or np.ndarray
        Calendar age(s), same shape as c14.

    Returns
    -------
    float or np.ndarray
        Corresponding d14c value(s).

    Notes
    -----
    - Supports element-wise operations.
    - This function is computed as the composition of `f14c_to_d14c` and `c14_to_f14c`:
        d14c = f14c_to_d14c(c14_to_f14c(c14),teta).
    """
    return f14c_to_d14c(c14_to_f14c(c14),teta)

def c14sig_to_d14csig(
    c14: NumberOrArray, 
    c14sig: NumberOrArray, 
    teta: NumberOrArray
) -> NumberOrArray:
    """
    Convert $^{14}$C uncertainty to d14c uncertainty.

    Parameters
    ----------
    c14 : float or np.ndarray
        Radiocarbon age(s).
    c14sig : float or np.ndarray
        Uncertainty in c14.
    teta : float or np.ndarray
        Calendar age(s), same shape as c14.

    Returns
    -------
    float or np.ndarray
        Corresponding uncertainty in d14c.

    Notes
    -----
    - Supports element-wise operations.
    - This function is computed as the composition of `f14csig_to_d14csig` and `c14sig_to_f14csig`:
        c14sig = f14csig_to_d14csig(
            c14sig_to_f14csig(c14,c14sig),
            teta
        ).
    """
    return f14csig_to_d14csig(
        c14sig_to_f14csig(c14,c14sig),
        teta
    )


# --------------------------
# Segment plotting
# --------------------------

# foncions pour tracer les segments / barres d'erreurs

def ajoute_segment_vertical(
    x: float,
    y_min: float,
    y_max: float,
    ax: plt.Axes = None,
    color: str = 'red',
    linestyle: str = '-',
    linewidth: float = 2,
    label: str = None,
    ticks: bool = True,
    tick_size: float = 0.1
) -> plt.Line2D:
    """
    Plot a vertical segment from (x, y_min) to (x, y_max) with optional horizontal ticks at ends.

    Parameters
    ----------
    x : float
        X-coordinate of the segment.
    y_min : float
        Starting Y-coordinate (bottom).
    y_max : float
        Ending Y-coordinate (top).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, uses current axis.
    color : str, optional
        Segment color (default 'red').
    linestyle : str, optional
        Line style (default '-').
    linewidth : float, optional
        Line width (default 2).
    label : str, optional
        Label for the segment.
    ticks : bool, optional
        If True, adds small horizontal ticks at the segment ends (default True).
    tick_size : float, optional
        Length of horizontal ticks (default 0.1).

    Returns
    -------
    matplotlib.lines.Line2D
        The main line object of the vertical segment.
    """
    if ax is None:
        ax = plt.gca()

    line = ax.plot([x, x], [y_min, y_max], color=color, linestyle=linestyle,
                   linewidth=linewidth, label=label)[0]

    if ticks:
        ax.plot([x - tick_size, x + tick_size], [y_min, y_min], color=color, linewidth=linewidth)
        ax.plot([x - tick_size, x + tick_size], [y_max, y_max], color=color, linewidth=linewidth)

    return line


def ajoute_segment_horizontal(
    y: float,
    x_min: float,
    x_max: float,
    ax: plt.Axes = None,
    color: str = 'blue',
    linestyle: str = '-',
    linewidth: float = 2,
    label: str = None,
    ticks: bool = True,
    tick_size: float = 0.1
) -> plt.Line2D:
    """
    Plot a horizontal segment from (x_min, y) to (x_max, y) with optional vertical ticks at ends.

    Parameters
    ----------
    y : float
        Y-coordinate of the segment.
    x_min : float
        Starting X-coordinate (left).
    x_max : float
        Ending X-coordinate (right).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, uses current axis.
    color : str, optional
        Segment color (default 'blue').
    linestyle : str, optional
        Line style (default '-').
    linewidth : float, optional
        Line width (default 2).
    label : str, optional
        Label for the segment.
    ticks : bool, optional
        If True, adds small vertical ticks at the segment ends (default True).
    tick_size : float, optional
        Length of vertical ticks (default 0.1).

    Returns
    -------
    matplotlib.lines.Line2D
        The main line object of the horizontal segment.
    """
    if ax is None:
        ax = plt.gca()

    line = ax.plot([x_min, x_max], [y, y], color=color, linestyle=linestyle,
                   linewidth=linewidth, label=label)[0]

    if ticks:
        ax.plot([x_min, x_min], [y - tick_size, y + tick_size], color=color, linewidth=linewidth)
        ax.plot([x_max, x_max], [y - tick_size, y + tick_size], color=color, linewidth=linewidth)

    return line


# --------------------------
# BP ↔ Calendar conversion
# --------------------------


def bp_to_calendar(bp: Union[int, np.ndarray]) -> Union[Tuple[int, str], np.ndarray]:
    """
    Converts BP (Before Present, ref. 1949) to calendar year BCE/CE.

    Parameters
    ----------
    bp : int or np.ndarray
        Year(s) in BP (Before Present, reference year 1949).

    Returns
    -------
    Tuple[int, str] or np.ndarray
        - If single int: tuple (year, 'BCE' or 'CE')
        - If numpy array: structured array of (year, era) for each element

    Notes
    -----
    - BP > 1949 corresponds to BCE, else CE.
    - Skips year 0: 1949 BP = 1 CE.
    """
    if isinstance(bp, np.ndarray):
        years = np.where(bp > 1949, bp - 1949, 1950 - bp)
        eras = np.where(bp > 1949, 'BCE', 'CE')
        return np.array(list(zip(years, eras)), dtype=object)
    else:
        if bp > 1949:
            year = bp - 1949
            return (year, 'BCE')
        else:
            year = 1950 - bp  # on saute l’an 0 → donc 1949 BP = 1 CE
            return (year, 'CE')



def calendar_to_bp(
    year: Union[int, np.ndarray],
    era: Union[str, np.ndarray]
) -> Union[int, np.ndarray]:
    """
    Converts calendar year(s) in BCE or CE to BP (Before Present, ref. 1949).

    Parameters
    ----------
    year : int or np.ndarray
        Positive calendar year(s) in BCE or CE.
    era : str or np.ndarray
        Era indicator(s): 'BCE' or 'CE'.
        If a NumPy array is provided, it must have the same shape as `year`.

    Returns
    -------
    int or np.ndarray
        Corresponding year(s) in BP.

    Raises
    ------
    ValueError
        If any element in `era` is not 'BCE' or 'CE', or if array shapes do not match.

    Notes
    -----
    - Skips year 0: 1 CE = 1949 BP.
    - Supports both scalar and vectorized inputs.
    - When given a structured array as returned by `bp_to_calendar`,
      it will internally extract the year and era columns before computation.

    Examples
    --------
    >>> calendar_to_bp(2500, 'BCE')
    4449

    >>> calendar_to_bp(2020, 'CE')
    -70

    >>> arr = np.array([[2500, 'BCE'], [2020, 'CE']], dtype=object)
    >>> calendar_to_bp(arr[:, 0].astype(int), arr[:, 1])
    array([4449.,  -70.])
    >>> calendar_to_bp(arr[:, 0], arr[:, 1])
    array([4449.,  -70.])
    """
    # Support structured array returned by bp_to_calendar
    if isinstance(year, np.ndarray) and year.dtype == object and year.ndim == 2 and year.shape[1] == 2:
        # form (year, era)
        year, era = year[:, 0].astype(int), year[:, 1]

    if isinstance(year, np.ndarray):
        if not isinstance(era, np.ndarray):
            raise ValueError("When 'year' is an array, 'era' must also be an array.")
        if year.shape != era.shape:
            raise ValueError("'year' and 'era' arrays must have the same shape.")

        bp = np.empty_like(year, dtype=float)
        mask_bce = era == 'BCE'
        mask_ce = era == 'CE'

        if not np.all(mask_bce | mask_ce):
            raise ValueError("All 'era' values must be either 'BCE' or 'CE'.")

        bp[mask_bce] = 1949 + year[mask_bce]
        bp[mask_ce] = 1950 - year[mask_ce]
        return bp
    else:
        if era == 'BCE':
            return 1949 + year
        elif era == 'CE':
            return 1950 - year  # on saute l’an 0
        else:
            raise ValueError("Era must be 'BCE' or 'CE'.")



__all__ = [
    "get_lib_data_paths",
    "load_data",
    "minimax_scaling",
    "minimax_scaling_reciproque",
    "d14c_to_f14c",
    "d14csig_to_f14csig",
    "f14c_to_d14c",
    "f14csig_to_d14csig",
    "f14c_to_c14",
    "f14csig_to_c14sig",
    "c14_to_f14c",
    "c14sig_to_f14csig",
    "d14c_to_c14",
    "d14csig_to_c14sig",
    "c14_to_d14c",
    "c14sig_to_d14csig",
    "ajoute_segment_vertical",
    "ajoute_segment_horizontal",
    "bp_to_calendar",
    "calendar_to_bp"
]

not_included_in_doc = [
    "read_params_from_file"
]