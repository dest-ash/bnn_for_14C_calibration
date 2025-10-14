# version de la librairie
__version__ = "0.1.0"


# fonctions de calibration :
#     - individuelle
#     - jointe
#     - par IntCal20

from .calibration import (
    individual_calibration,
    IntCal20_calibration,
    joint_calibration
)

# fonctions d'affichage :
#     - résultats de calibration
#     - courbes de calibration

from .calib_plot_functions import (
    plot_calib_results,
    plot_individual_calibration_curve_part_1,
    plot_individual_calibration_curve_part_2,
    plot_bnn_calibration_curve,
    plot_IntCal20_curve
)

# fonctions de gestion du cache local
#     - téléchargement
#     - suppression

from .manage_cache import(
    download_cache_lib_data,
    clear_cache
)

__all__ = [
    # fonctions de calibration
    "individual_calibration", 
    "IntCal20_calibration", 
    "joint_calibration", 

    # fonctions d'affichage
    "plot_calib_results", 
    "plot_individual_calibration_curve_part_1", 
    "plot_individual_calibration_curve_part_2", 
    "plot_bnn_calibration_curve", 
    "plot_IntCal20_curve",

    # fonctions de gestion du cache local
    "download_cache_lib_data",
    "clear_cache"
]