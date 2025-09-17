# -*- coding: utf-8 -*-


import requests
from pathlib import Path
import time
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
    repo_api_url: str,
    local_dir: Path,
    token: str = None,
    retries: int = 3,
    timeout: float = 10,
    sleep_time: float = 0.2
):
    """
    Télécharge un dossier GitHub (public ou privé) en gérant fichiers classiques et LFS.
    Détecte automatiquement la branche par défaut et retente les fichiers LFS échoués.
    
    Parameters:
    - repo_api_url: API URL du dossier (ex: https://api.github.com/repos/{owner}/{repo}/contents/{path})
    - local_dir: chemin local où sauvegarder les fichiers
    - token: optionnel, nécessaire pour repos privés ou pour éviter limites API
    - retries: nombre de retentatives pour les fichiers LFS échoués
    - timeout: délai maximal pour chaque requête HTTP
    - sleep_time: délai entre chaque téléchargement pour éviter surcharge serveur
    """
    headers = {"Authorization": f"token {token}"} if token else {}

    # Déterminer la branche par défaut
    parts = repo_api_url.split('/')
    owner, repo = parts[3], parts[4]
    repo_info_url = f"https://api.github.com/repos/{owner}/{repo}"
    repo_info = requests.get(repo_info_url, headers=headers, timeout=timeout).json()
    default_branch = repo_info.get("default_branch", "main")

    failed_lfs_files = []

    def _download_folder(api_url, local_dir):
        response = requests.get(api_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        items = response.json()

        for item in items:
            if item["type"] == "file":
                file_url = item.get("download_url")
                local_path = local_dir / item["name"]
                local_path.parent.mkdir(parents=True, exist_ok=True)

                if not file_url:
                    print(f"Skipping {item['name']}, no download URL available")
                    continue

                print(f"Downloading {file_url} → {local_path}")
                r = requests.get(file_url, headers=headers, timeout=timeout)
                r.raise_for_status()

                if r.text.startswith("version https://git-lfs.github.com/spec/v1"):
                    # Fichier LFS
                    lines = r.text.splitlines()
                    oid_line = next((l for l in lines if l.startswith("oid sha256:")), None)
                    if oid_line:
                        oid = oid_line.split("oid sha256:")[1]
                        # URL brute correcte pour n'importe quelle branche
                        lfs_url = f"https://github.com/{owner}/{repo}/raw/{default_branch}/{item['path']}"
                        try:
                            lfs_r = requests.get(lfs_url, headers=headers, timeout=timeout)
                            if lfs_r.status_code in (403, 404):
                                print(f"    ❌ LFS file {item['name']} unavailable (status {lfs_r.status_code}). Will retry later.")
                                failed_lfs_files.append((lfs_url, local_path))
                            else:
                                lfs_r.raise_for_status()
                                local_path.write_bytes(lfs_r.content)
                        except requests.RequestException as e:
                            print(f"    ❌ Cannot download LFS file {item['name']}: {e}")
                            failed_lfs_files.append((lfs_url, local_path))
                    else:
                        print(f"    ❌ Invalid LFS pointer for {item['name']}, skipping.")
                else:
                    local_path.write_bytes(r.content)

                time.sleep(sleep_time)

            elif item["type"] == "dir":
                subdir = local_dir / item["name"]
                subdir.mkdir(parents=True, exist_ok=True)
                _download_folder(item["url"], subdir)

    # Première passe
    _download_folder(repo_api_url, local_dir)

    # Retenter les fichiers LFS échoués
    attempt = 1
    while failed_lfs_files and attempt <= retries:
        print(f"\nRetrying LFS files, attempt {attempt}/{retries}...")
        remaining = []
        for lfs_url, local_path in failed_lfs_files:
            try:
                lfs_r = requests.get(lfs_url, headers=headers, timeout=timeout)
                if lfs_r.status_code in (403, 404):
                    print(f"    ❌ Still unavailable: {local_path.name} (status {lfs_r.status_code})")
                    remaining.append((lfs_url, local_path))
                else:
                    lfs_r.raise_for_status()
                    local_path.write_bytes(lfs_r.content)
                    print(f"    ✅ Downloaded {local_path.name}")
            except requests.RequestException as e:
                print(f"    ❌ Error downloading {local_path.name}: {e}")
                remaining.append((lfs_url, local_path))
            time.sleep(sleep_time)
        failed_lfs_files = remaining
        attempt += 1

    if failed_lfs_files:
        print("\n⚠️ Some LFS files could not be downloaded after retries:")
        for _, local_path in failed_lfs_files:
            print(f" - {local_path}")
    else:
        print("\n✅ All files downloaded successfully!")




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

