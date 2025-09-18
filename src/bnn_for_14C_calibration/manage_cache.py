# -*- coding: utf-8 -*-


import requests
from pathlib import Path
import time
import json
import re
import gdown
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


def is_google_drive_url(url: str) -> bool:
    """
    Vérifie si une URL correspond à un fichier ou dossier Google Drive.
    
    Paramètres
    ----------
    url : str
        L'URL à tester.

    Retour
    ------
    bool
        True si l'URL contient 'drive.google.com', False sinon.
    """
    return "drive.google.com" in url


def extract_drive_file_id(url: str) -> str:
    """
    Extrait l'ID d'un fichier Google Drive à partir d'une URL publique.
    
    Paramètres
    ----------
    url : str
        L'URL publique du fichier Google Drive.
        Exemples :
            - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
            - https://drive.google.com/open?id=FILE_ID

    Retour
    ------
    str ou None
        L'ID du fichier Google Drive, ou None si l'ID n'a pas pu être extrait.
    """
    m = re.search(r"/file/d/([^/]+)", url)
    if m:
        return m.group(1)
    m2 = re.search(r"[?&]id=([^&]+)", url)
    if m2:
        return m2.group(1)
    return None


def download_from_google_drive(url_or_id: str, output_path: Path, sleep_time: float = 0.2):
    """
    Télécharge un fichier ou un dossier depuis Google Drive.
    Si l'URL correspond à un dossier, utilise gdown.download_folder.

    Paramètres
    ----------
    url_or_id : str
        L'URL publique Google Drive ou l'ID du fichier/dossier.
    output_path : Path
        Le chemin local où sauvegarder le fichier ou dossier téléchargé.
    sleep_time : float, optional (default=0.2)
        Temps en secondes à attendre après chaque téléchargement pour limiter les requêtes.
    
    Comportement
    ------------
    - Crée les dossiers parents si nécessaire.
    - Gère les erreurs d'accès et les affiche sans interrompre le script.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if "drive.google.com/drive/folders" in url_or_id:
            print(f"Downloading Google Drive folder {url_or_id} → {output_path}")
            gdown.download_folder(url_or_id, output=str(output_path), quiet=False)
        else:
            file_id = extract_drive_file_id(url_or_id) or url_or_id
            print(f"Downloading Google Drive file id {file_id} → {output_path}")
            gdown.download(id=file_id, output=str(output_path), quiet=False, fuzzy=True)
    except Exception as e:
        print(f"    ❌ Erreur Google Drive : {e}")
    time.sleep(sleep_time)


def download_github_with_drive_map(
    api_url: str,
    local_dir: Path,
    token: str = None,
    timeout: float = 10,
    sleep_time: float = 0.2
):
    """
    Télécharge le contenu d'un dossier GitHub, en remplaçant certains fichiers
    par leur équivalent Google Drive selon un fichier `drive_map.json` local
    à chaque sous-dossier.

    Paramètres
    ----------
    api_url : str
        URL de l'API GitHub pour accéder au contenu du dossier.
        Exemple : https://api.github.com/repos/username/repo/contents/path
    local_dir : Path
        Répertoire local où sauvegarder les fichiers téléchargés.
    token : str, optional
        Token d'authentification GitHub (nécessaire même pour certains repos publics si quotas dépassés).
    timeout : float, optional (default=10)
        Timeout en secondes pour chaque requête HTTP.
    sleep_time : float, optional (default=0.2)
        Temps en secondes à attendre entre les téléchargements pour éviter de saturer le serveur.

    Comportement
    ------------
    - Crée la structure de dossiers correspondante localement.
    - Cherche un fichier `drive_map.json` dans chaque sous-dossier et télécharge les fichiers correspondants depuis Google Drive.
    - Les autres fichiers GitHub sont téléchargés normalement.
    - Les sous-dossiers sont parcourus récursivement.
    - Ne gère pas Git LFS, car tous les fichiers volumineux suivis par Git LFS sont supposés être mapés dans Drive. Ceci permet 
        de contourner les limites de bande passante en téléchargement inhérentes à Git LFS pour un compte gratuit.
    
    Exemple
    -------
    >>> api_url = "https://api.github.com/repos/monuser/monrepo/contents/data"
    >>> download_github_with_drive_map(api_url, Path("local_data"))
    """
    headers = {"Authorization": f"token {token}"} if token else {}

    def get_default_branch(owner: str, repo: str) -> str:
        """
        Récupère la branche par défaut du dépôt GitHub.
        
        Paramètres
        ----------
        owner : str
            Nom du propriétaire du dépôt.
        repo : str
            Nom du dépôt.
        
        Retour
        ------
        str
            Nom de la branche par défaut (ex: "main" ou "master").
        """
        repo_info_url = f"https://api.github.com/repos/{owner}/{repo}"
        r_info = requests.get(repo_info_url, headers=headers, timeout=timeout)
        r_info.raise_for_status()
        return r_info.json().get("default_branch", "main")

    def _download_folder(api_url: str, local_dir: Path):
        """
        Fonction interne récursive pour télécharger un dossier GitHub
        et remplacer les fichiers selon drive_map.json.
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        drive_map_path = local_dir / "drive_map.json"
        drive_map = {}
        if drive_map_path.exists():
            with open(drive_map_path, "r", encoding="utf-8") as f:
                drive_map = json.load(f)

        response = requests.get(api_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        items = response.json()

        parts = api_url.split('/')
        owner, repo = parts[3], parts[4]
        default_branch = get_default_branch(owner, repo)

        for item in items:
            relative_name = item["name"]
            local_path = local_dir / relative_name

            # Cas Google Drive selon drive_map.json
            if relative_name in drive_map and is_google_drive_url(drive_map[relative_name]):
                download_from_google_drive(drive_map[relative_name], local_path, sleep_time=sleep_time)
                continue

            # Cas fichier GitHub classique
            if item["type"] == "file":
                file_url = item.get("download_url")
                if not file_url:
                    print(f"Skipping {item['name']}, no download URL")
                    continue
                print(f"Downloading GitHub file {file_url} → {local_path}")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                r = requests.get(file_url, headers=headers, timeout=timeout)
                r.raise_for_status()
                local_path.write_bytes(r.content)
                time.sleep(sleep_time)

            # Cas sous-dossier
            elif item["type"] == "dir":
                subdir = local_dir / relative_name
                _download_folder(item["url"], subdir)

    _download_folder(api_url, local_dir)





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
    if overwrite or not (CACHE_DIR.exists() and CACHE_DIR.is_dir()) :
        if overwrite :
            print(f"""
                overwrite is {overwrite} : the cache will be cleared before 
                downloading it again...
            """)
            clear_cache()
        print(f"Creating cache directory at: {CACHE_DIR}")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR_LOCAL.mkdir(exist_ok=True)
        download_github_with_drive_map(MODELS_DIR_API_URL, MODELS_DIR_LOCAL)
        print(f"""
        ✅ Cache directory created at: {CACHE_DIR}, and filled with all the 
        contents of the 'models' directory downloaded from GitHub and Google Drive.
        """)
    else :
        print(f"""
            An existing cache directory is located at {CACHE_DIR}  and overwrite is {overwrite}.
            If you wish to force the cache download, set overwrite to True.
        """)

if __name__ == "__main__":
    download_cache_lib_data()

