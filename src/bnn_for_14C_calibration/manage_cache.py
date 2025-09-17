# -*- coding: utf-8 -*-

from pathlib import Path


# URL pour accéder aux données distantes à mettre en cache
CACHE_URL = "https://raw.githubusercontent.com/dest-ash/bnn_for_14C_calibration/main/models/"

def download_cache_lib_data(
    overwrite = False
):
    url = "https://github.com/dest-ash/bnn_for_14C_calibration/tree/main/models"
    cache_dir_name = ".bnn_for_14C_calibration"
    path_to_user_home = Path.home().resolve()

