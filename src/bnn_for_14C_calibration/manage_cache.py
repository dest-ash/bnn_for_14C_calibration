# -*- coding: utf-8 -*-


import requests
from pathlib import Path
import time
import json
import re
import gdown
import zipfile
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


def clear_cache():
    """
    Supprime complètement le dossier cache de la librairie.
    """
    if CACHE_DIR.exists():
        print(f"🗑️ removing cache directory at : {CACHE_DIR}...")
        shutil.rmtree(CACHE_DIR)
        print(f"🗑️ cache removed!")
    else:
        print("ℹ️ No existing cache!")


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
            print(f"📂 Downloading Google Drive folder {url_or_id} → {output_path}")
            gdown.download_folder(url_or_id, output=str(output_path), quiet=False)
        else:
            file_id = extract_drive_file_id(url_or_id) or url_or_id
            print(f"📄 Downloading Google Drive file id {file_id} → {output_path}")
            gdown.download(id=file_id, output=str(output_path), quiet=False, fuzzy=True)
    except Exception as e:
        raise RuntimeError(f"Google Drive download failed for {output_path}: {e}")
    time.sleep(sleep_time)


def download_from_huggingface(url: str, output_path: Path, timeout: int = 10, sleep_time: float = 0.2):
    """
    Download a file from Hugging Face Hub.
    If the file is a .zip archive, its contents are extracted directly into `output_path`
    without preserving the top-level folder from the archive.

    Parameters
    ----------
    url : str
        The direct Hugging Face Hub URL to the file.
    output_path : pathlib.Path
        Local path where the file or extracted contents will be saved.
    timeout : float, optional
        Timeout for HTTP requests (default 10 seconds).
    sleep_time : float, optional
        Delay in seconds after download to avoid throttling (default 0.2).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = output_path.with_suffix(".tmp")

    print(f"⬇️ Downloading from Hugging Face: {url} → {output_path}")
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(temp_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        time.sleep(sleep_time)

        # Dézippe automatiquement si c'est un zip
        if str(url).endswith(".zip"):
            print(f"📦 Extracting zip {temp_file} → {output_path} (flatten top-level folder)")
            with zipfile.ZipFile(temp_file, "r") as zip_ref:
                for member in zip_ref.infolist():
                    # Split path and skip the first component (top-level folder)
                    path_parts = member.filename.split('/')
                    if len(path_parts) > 1:
                        target_path = output_path.joinpath(*path_parts[1:])
                    else:
                        target_path = output_path / member.filename

                    if member.is_dir():
                        target_path.mkdir(parents=True, exist_ok=True)
                    else:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        with zip_ref.open(member) as source, open(target_path, "wb") as target:
                            target.write(source.read())
            temp_file.unlink()  # Remove temporary zip file after extraction
        else:
            temp_file.rename(output_path)

    except Exception as e:
        raise RuntimeError(f"Hugging Face download failed for {output_path}: {e}")


def download_github_with_drive_map(
    api_url: str,
    local_dir: Path,
    token: str = None,
    timeout: float = 10,
    sleep_time: float = 0.2
):
    """
    Download a GitHub folder while integrating external sources via drive_map.json.

    Parameters
    ----------
    api_url : str
        GitHub API URL pointing to the folder to download.
        Example: https://api.github.com/repos/username/repo/contents/models
    local_dir : pathlib.Path
        Local directory where the contents will be downloaded.
    token : str, optional
        GitHub personal access token if needed (default None).
    timeout : float, optional
        Timeout in seconds for HTTP requests (default 10).
    sleep_time : float, optional
        Delay in seconds between downloads to avoid throttling (default 0.2).

    Raises
    ------
    RuntimeError
        If a file or folder cannot be downloaded from both Hugging Face and Google Drive.
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

        # Récupère la liste des fichiers du dossier GitHub
        response = requests.get(api_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        items = response.json()

        # Charge drive_map.json depuis GitHub si présent
        drive_map = {}
        for item in items:
            if item["type"] == "file" and item["name"] == "drive_map.json":
                local_map_path = local_dir / "drive_map.json"
                print(f"📄 Found drive_map.json in {api_url}, downloading → {local_map_path}")
                r_map = requests.get(item["download_url"], headers=headers, timeout=timeout)
                r_map.raise_for_status()
                local_map_path.write_text(r_map.text, encoding="utf-8")  # sauvegarde en local
                drive_map = json.loads(r_map.text)  # charge en mémoire
                break

        # Extrait owner/repo pour GitHub
        parts = api_url.split('/')
        try:
            i = parts.index("repos")
            owner, repo = parts[i+1], parts[i+2]
        except (ValueError, IndexError):
            raise ValueError(f"Cannot extract owner/repo from GitHub API URL: {api_url}")

        default_branch = get_default_branch(owner, repo)

        for item in items:
            relative_name = item["name"]
            local_path = local_dir / relative_name

            # Gestion des fichiers et dossiers mappés via drive_map.json
            if relative_name in drive_map:
                mapped = drive_map[relative_name]
                success = False

                # Hugging Face first
                if "huggingface" in mapped:
                    try:
                        download_from_huggingface(mapped["huggingface"], local_path,
                                                  timeout=timeout, sleep_time=sleep_time)
                        success = True
                    except Exception as e_hf:
                        print(f"⚠️ Hugging Face download failed for {relative_name}: {e_hf}")

                # Google Drive fallback
                if not success and "drive" in mapped:
                    try:
                        download_from_google_drive(mapped["drive"], local_path, sleep_time=sleep_time)
                        success = True
                    except Exception as e_drive:
                        print(f"⚠️ Google Drive download failed for {relative_name}: {e_drive}")

                if not success:
                    # suppression du cache partiel éventuellement  téléchargé
                    print("""
                    ⚠️ Failed to download all the data to cache. Removing the created cache before raising a 
                        RuntimeError with more details about the file or the folder that matters
                    """)
                    clear_cache()
                    raise RuntimeError(f"❌ Failed to download {relative_name} from both Hugging Face and Google Drive")

                # ✅ Already downloaded via external sources, skip GitHub
                continue

            # Cas fichier GitHub classique
            if item["type"] == "file":
                if relative_name in drive_map:
                    # ✅ fichier déjà pris en charge via Hugging Face ou Google Drive
                    print(f"Skipping GitHub file {relative_name}, handled via Hugging Face or Google Drive")
                    continue
                if item["name"] == "drive_map.json":
                    # déjà téléchargé ci-dessus
                    continue
                file_url = item.get("download_url")
                if not file_url:
                    print(f"Skipping {item['name']}, no download URL")
                    continue
                print(f"⬇️ Downloading GitHub file {file_url} → {local_path}")
                local_path.parent.mkdir(parents=True, exist_ok=True)
                r = requests.get(file_url, headers=headers, timeout=timeout)
                r.raise_for_status()
                local_path.write_bytes(r.content)
                time.sleep(sleep_time)

            # Cas sous-dossier GitHub
            elif item["type"] == "dir":
                if relative_name in drive_map:
                    # ✅ dossier déjà pris en charge via Hugging Face ou Google Drive
                    print(f"Skipping GitHub folder {relative_name}, handled via Hugging Face or Google Drive")
                    continue
                subdir = local_dir / relative_name
                _download_folder(item["url"], subdir)

    _download_folder(api_url, local_dir)




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
        else :
            print("""
                This may be the first time you need package functions that use 
                local cache data to work. A local cache directory is going to be 
                created and will be used after if needed without a new downloading
                unless you delete the cache. To download this cache, network connexion 
                must be available ; also some disk space is required (less than 1 GB at all).
            """)
        print(f"******************** Creating cache directory at: {CACHE_DIR} ********************")
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


# fonctions publiques
__all__ = [
    # fonctions de gestion du cache local
    "download_cache_lib_data",
    "clear_cache"
]

if __name__ == "__main__":
    download_cache_lib_data()

