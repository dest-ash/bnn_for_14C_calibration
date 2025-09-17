# -*- coding: utf-8 -*-


from pathlib import Path
import requests
import shutil


# ========================================================================
# définition des chemins constants vers le cache local 
# ========================================================================

# chemin de sauvegarde pour le cache local
CACHE_DIR_NAME = ".bnn_for_14C_calibration"
CACHE_DIR = Path.home().resolve() / CACHE_DIR_NAME

# API GitHub pour lister le contenu le dossier 'models' destiné au cache local
# et chemin pour le dossier  distant 'models' dans le cache local
MODELS_DIR_API_URL = "https://api.github.com/repos/dest-ash/bnn_for_14C_calibration/contents/models"
MODELS_DIR_LOCAL = CACHE_DIR / "models"

# ========================================================================
# fonctions de téléchargement des données et leur mise en cache
# ========================================================================


# téléchargement d'un dossier github
def download_github_folder(
    api_url: str, 
    local_dir: Path
):
    response = requests.get(api_url)
    response.raise_for_status()
    items = response.json()

    for item in items:
        if item["type"] == "file":
            file_url = item["download_url"]
            local_path = local_dir / item["name"]
            print(f"Downloading {file_url} → {local_path}")
            r = requests.get(file_url)
            r.raise_for_status()
            local_path.write_bytes(r.content)
        elif item["type"] == "dir":
            subdir = local_dir / item["name"]
            subdir.mkdir(exist_ok=True)
            download_github_folder(item["url"], subdir)


def clear_cache():
    """
    Supprime complètement le dossier cache de la librairie.
    """
    if CACHE_DIR.exists():
        print(f"🗑️ removing cache directory at : {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR)
        print(f"🗑️ cache removed!")
    else:
        print("ℹ️ No existing cache!")

def download_cache_lib_data(
    overwrite = False
):
    if overwrite or not (CACHE_DIR.exists() and CACHE_DIR.is_dir())
        print(f"Creating cache directory at: {CACHE_DIR}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR_LOCAL.mkdir(exist_ok=True)
        download_github_folder(MODELS_DIR_API_URL, MODELS_DIR_LOCAL)
        print(f"""
        ✅ Cache directory created at: {CACHE_DIR}, and filled with all 
        the contents of the 'models' directory downloaded from GitHub.
        """)
    else :
        print(f"""
            An existing cache directory is located at {CACHE_DIR}  and overwrite is {overwrite}.
            If you wish to force the cache download, set overwrite to True.
        """)

if __name__ == "__main__":
    download_cache_lib_data()

