# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def download_cache_lib_data(
    overwrite = False
):
    url = "https://github.com/dest-ash/bnn_for_14C_calibration/tree/main/models"
    cache_dir_name = ".bnn_for_14C_calibration"
    path_to_user_home = Path.home().resolve()



def read_params_from_file(file_path) :
    """

    Parameters
    ----------
    file_path : TYPE str
        DESCRIPTION. le chemin vers le fichier contenant les paramètres

    Returns
    -------
    keys : TYPE Dict.
        dictionnaire qui contiendra les paramètres sous format "nom : valeur"

    """
    keys = {}
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
                    if __name__ == "__main__" :
                        print(f"""
                              Erreur ignorée : {e} \n
                              Si le paramètre {key} est censé être une chaîne de caractère, c'est le comportement attendu. \n
                              Sinon, c'est un cas non prévu et le paramètre sera enregistré comme chaîne de caractère.
                        """)
                    pass
            
            keys[key] = value
    return keys

"""
Fonction de chargement des données dans des dataframes pandas
"""
def load_data(path, sep =";") :
    dataset = pd.read_csv(path, sep =sep)
    return dataset


"""
Transformation minimax manuelle
"""
def minimax_scaling(x,Max,Min) :
    return (x-Min)/(Max-Min)

"""
Transformation réciproque minimax manuelle
"""
def minimax_scaling_reciproque(x,Max,Min) :
    return (Max-Min)*x + Min


"""
Relations entre les différents domaines du C14

# c14 = domaine des âges C14
# f14c = domaine d'acquisition des mesures C14 (i.e. le rapport isotopique C14/C12)
# d14c = delta c14
# teta = date (age calendaire/calibré : calage)

# à chaque domaine est associée l'incertitude de laboratoire : les formules de sa conversion 
# d'un domaine à l'autre découlent de la delta-méthode (par exemple)
"""

# domaine d14c vers domaine f14c

def d14c_to_f14c(d14c, teta) :
    f14c = (1/1000*d14c + 1)*np.exp(-teta/8267)
    return f14c

def d14csig_to_f14csig(d14csig, teta) :
    f14csig = d14csig*np.exp(-teta/8267)/1000
    return f14csig


# domaine f14c vers domaine d14c

def f14c_to_d14c(f14c, teta) :
    d14c = 1000*(-1 + f14c*np.exp(teta/8267))
    return d14c

def f14csig_to_d14csig(f14csig, teta) :
    d14csig = 1000*f14csig*np.exp(teta/8267)
    return d14csig


# domaine f14c vers domaine c14

def f14c_to_c14(f14c) :
    c14 = -8033*np.log(f14c)
    return c14

def f14csig_to_c14sig(f14c, f14csig) :
    # fonction basée sur la delta-méthode
    c14_sig = f14csig*8033/f14c # f14c > 0
    return c14_sig


# domaine c14 vers domaine f14c

def c14_to_f14c(c14) :
    f14c = np.exp(-c14/8033)
    return f14c

def c14sig_to_f14csig(c14, c14sig) :
    # fonction basée sur la delta-méthode
    f14c = c14_to_f14c(c14 = c14)
    f14c_sig = c14sig*f14c/8033 # f14c > 0
    return f14c_sig


# domaine d14c vers domaine c14

def d14c_to_c14(d14c, teta) :
  return f14c_to_c14(d14c_to_f14c(d14c,teta))

def d14csig_to_c14sig(d14c, d14csig, teta) :
  return f14csig_to_c14sig(
    d14c_to_f14c(d14c,teta),
    d14csig_to_f14csig(d14csig,teta)
  )


# domaine c14 vers domaine d14c

def c14_to_d14c(c14, teta) :
  return f14c_to_d14c(c14_to_f14c(c14),teta)

def c14sig_to_d14csig(c14, c14sig, teta) :
  return f14csig_to_d14csig(
    c14sig_to_f14csig(c14,c14sig),
    teta
  )


# foncions pour tracer les segments / barres d'erreurs

def ajoute_segment_vertical(x, y_min, y_max, ax=None, color='red', linestyle='-', linewidth=2, label=None,
                            ticks=True, tick_size=0.1):
    """
    Trace un segment vertical de (x, y_min) à (x, y_max) et ajoute, si désiré, des traits horizontaux aux extrémités.

    Paramètres
    ----------
    x : float
        L'abscisse du segment.
    y_min : float
        Ordonnée de départ du segment (bas).
    y_max : float
        Ordonnée de fin du segment (haut).
    ax : matplotlib.axes.Axes, optional
        L'axe sur lequel tracer. Si None, utilise l'axe courant.
    color : str, optional
        Couleur du segment.
    linestyle : str, optional
        Style de trait.
    linewidth : float, optional
        Épaisseur du segment.
    label : str, optional
        Légende du segment.
    ticks : bool, optional
        Si True, ajoute des petits traits horizontaux aux extrémités.
    tick_size : float, optional
        Longueur des petits traits horizontaux.

    Retourne
    --------
    line : matplotlib.lines.Line2D
        L'objet ligne du segment principal.
    """
    if ax is None:
        ax = plt.gca()

    line = ax.plot([x, x], [y_min, y_max], color=color, linestyle=linestyle,
                   linewidth=linewidth, label=label)[0]

    if ticks:
        ax.plot([x - tick_size, x + tick_size], [y_min, y_min], color=color, linewidth=linewidth)
        ax.plot([x - tick_size, x + tick_size], [y_max, y_max], color=color, linewidth=linewidth)

    return line


def ajoute_segment_horizontal(y, x_min, x_max, ax=None, color='blue', linestyle='-', linewidth=2, label=None,
                              ticks=True, tick_size=0.1):
    """
    Trace un segment horizontal de (x_min, y) à (x_max, y) et ajoute, si désiré, des traits verticaux aux extrémités.

    Paramètres
    ----------
    y : float
        L'ordonnée du segment.
    x_min : float
        Abscisse de départ du segment (gauche).
    x_max : float
        Abscisse de fin du segment (droite).
    ax : matplotlib.axes.Axes, optional
        L'axe sur lequel tracer. Si None, utilise l'axe courant.
    color : str, optional
        Couleur du segment.
    linestyle : str, optional
        Style de trait.
    linewidth : float, optional
        Épaisseur du segment.
    label : str, optional
        Légende du segment.
    ticks : bool, optional
        Si True, ajoute des petits traits verticaux aux extrémités.
    tick_size : float, optional
        Longueur des petits traits verticaux.

    Retourne
    --------
    line : matplotlib.lines.Line2D
        L'objet ligne du segment principal.
    """
    if ax is None:
        ax = plt.gca()

    line = ax.plot([x_min, x_max], [y, y], color=color, linestyle=linestyle,
                   linewidth=linewidth, label=label)[0]

    if ticks:
        ax.plot([x_min, x_min], [y - tick_size, y + tick_size], color=color, linewidth=linewidth)
        ax.plot([x_max, x_max], [y - tick_size, y + tick_size], color=color, linewidth=linewidth)

    return line



def bp_to_calendar(bp):
    """
    Convertit une date BP (Before Present, référence 1949) en BCE/CE.
    Retourne un tuple : (année, 'BCE' ou 'CE')
    """
    if bp > 1949:
        year = bp - 1949
        return (year, 'BCE')
    else:
        year = 1950 - bp  # on saute l’an 0 → donc 1949 BP = 1 CE
        return (year, 'CE')


def calendar_to_bp(year, era):
    """
    Convertit une date en BCE ou CE vers BP.
    year : entier positif
    era  : 'BCE' ou 'CE'
    """
    if era == 'BCE':
        return 1949 + year
    elif era == 'CE':
        return 1950 - year  # on saute l’an 0
    else:
        raise ValueError("L'ère doit être 'BCE' ou 'CE'.")

