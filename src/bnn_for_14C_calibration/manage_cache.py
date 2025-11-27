# -*- coding: utf-8 -*-


import requests
from pathlib import Path
import time
import json
import re
import gdown
import zipfile
import shutil

from typing import (
    Optional, 
    Union, 
    # List, 
    Dict, 
    # Tuple, 
    Any
)

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


def clear_cache() -> None:
    """
    Remove the entire local cache directory used by the library.

    This function deletes the cache directory and all its contents from
    the local filesystem. If the cache directory does not exist, a message
    is printed to indicate that there is no existing cache to remove.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - The cache directory path is set to `/home/user/.bnn_for_14C_calibration`.
    - This operation is irreversible; all cached data will be lost.
    - The cache data can be downloaded again by using the function `download_cache_lib_data`.

    Examples
    --------
    >>> clear_cache()
    🗑️ removing cache directory at: /home/user/.bnn_for_14C_calibration...
    🗑️ cache removed!
    """
    if CACHE_DIR.exists():
        print(f"🗑️ removing cache directory at: {CACHE_DIR}...")
        shutil.rmtree(CACHE_DIR)
        print(f"🗑️ cache removed!")
    else:
        print("ℹ️ No existing cache!")


def is_google_drive_url(url: str) -> bool:
    """
    Check whether a given URL corresponds to a Google Drive file or folder.

    Parameters
    ----------
    url : str
        The URL to check.

    Returns
    -------
    bool
        True if the URL contains 'drive.google.com', otherwise False.

    Examples
    --------
    >>> is_google_drive_url("https://drive.google.com/file/d/12345/view?usp=sharing")
    True
    >>> is_google_drive_url("https://example.com/file.txt")
    False
    """
    return "drive.google.com" in url


def extract_drive_file_id(url: str) -> Union[str, None]:
    """
    Extract the Google Drive file ID from a public URL.

    Parameters
    ----------
    url : str
        The public Google Drive file URL.  
        Examples include:

        - https://drive.google.com/file/d/FILE_ID/view?usp=sharing
        - https://drive.google.com/open?id=FILE_ID

    Returns
    -------
    str or None
        The extracted Google Drive file ID, or None if it could not be found.

    Examples
    --------
    >>> extract_drive_file_id("https://drive.google.com/file/d/ABC123/view")
    'ABC123'
    >>> extract_drive_file_id("https://example.com/file.txt")
    None
    """
    m = re.search(r"/file/d/([^/]+)", url)
    if m:
        return m.group(1)
    m2 = re.search(r"[?&]id=([^&]+)", url)
    if m2:
        return m2.group(1)
    return None


def download_from_google_drive(
    url_or_id: str,
    output_path: Path,
    sleep_time: float = 0.2
) -> None:
    """
    Download a file or folder from Google Drive.

    If the provided URL or ID refers to a folder, this function uses
    `gdown.download_folder`; otherwise, it downloads a single file.

    Parameters
    ----------
    url_or_id : str
        The public Google Drive URL or file/folder ID.
    output_path : Path
        The local path where the downloaded file or folder will be saved.
    sleep_time : float, optional
        Time in seconds to wait after the download (default is 0.2 seconds).

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the download fails for any reason.

    Notes
    -----
    - Parent directories will be created automatically if they do not exist.
    - Uses the `gdown` library for downloading from Google Drive.

    Examples
    --------
    >>> download_from_google_drive("https://drive.google.com/file/d/ABC123/view", Path("output.zip"))
    📄 Downloading Google Drive file id ABC123 → output.zip
    🗂️ Download complete!
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


def download_from_huggingface(
    url: str,
    output_path: Path,
    timeout: int = 10,
    sleep_time: float = 0.2
) -> None:
    """
    Download a file from the Hugging Face Hub.

    If the downloaded file is a `.zip` archive, its contents are automatically
    extracted directly into `output_path` without preserving the top-level
    folder contained in the archive.

    Parameters
    ----------
    url : str
        The direct Hugging Face Hub URL to the file.
    output_path : Path
        The local path where the file or extracted contents will be saved.
    timeout : int, optional
        Timeout for HTTP requests, in seconds (default is 10).
    sleep_time : float, optional
        Delay in seconds after the download to prevent throttling (default is 0.2).

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the download or extraction process fails.

    Notes
    -----
    - Parent directories are automatically created if they do not exist.
    - Temporary `.tmp` files are used during download to ensure file integrity.
    - `.zip` archives are extracted with flattened top-level directories.

    Examples
    --------
    >>> download_from_huggingface(
    ...     "https://huggingface.co/username/model/resolve/main/model.zip",
    ...     Path("models/model")
    ... )
    ⬇️ Downloading from Hugging Face: https://huggingface.co/username/model/resolve/main/model.zip → models/model
    📦 Extracting zip models/model.tmp → models/model (flatten top-level folder)
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
    token: Optional[str] = None,
    timeout: float = 10,
    sleep_time: float = 0.2
) -> None:
    """
    Download a GitHub folder and integrate external sources defined in a `drive_map.json` file.

    This function retrieves a folder from a GitHub repository via the GitHub API.
    If a `drive_map.json` file is present, it will use the mapping defined in it
    to download certain files or subfolders from Hugging Face or Google Drive instead
    of GitHub.

    Parameters
    ----------
    api_url : str
        GitHub API URL pointing to the folder to download.  
        Example: ``https://api.github.com/repos/username/repo/contents/models``
    local_dir : Path
        Local directory where the contents will be downloaded.
    token : str, optional
        GitHub personal access token, if authentication is required (default is None).
    timeout : float, optional
        Timeout in seconds for HTTP requests (default is 10).
    sleep_time : float, optional
        Delay in seconds between downloads to prevent rate limiting (default is 0.2).

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If a file or folder cannot be downloaded from both Hugging Face and Google Drive.
    ValueError
        If the GitHub API URL is invalid or the repository owner/repo cannot be extracted.

    Notes
    -----
    - Creates parent directories automatically if they do not exist.
    - Uses Hugging Face and Google Drive as fallback sources when specified in
      ``drive_map.json``.
    - Partially downloaded cache is cleared if an unrecoverable error occurs.

    Examples
    --------
    >>> download_github_with_drive_map(
    ...     "https://api.github.com/repos/dest-ash/bnn_for_14C_calibration/contents/models",
    ...     Path.home() / ".bnn_for_14C_calibration" / "models"
    ... )
    📄 Found drive_map.json in https://api.github.com/repos/dest-ash/bnn_for_14C_calibration/contents/models, downloading → ~/.bnn_for_14C_calibration/models/drive_map.json
    ⬇️ Downloading GitHub file https://raw.githubusercontent.com/... → ~/.bnn_for_14C_calibration/models/file.pt
    """

    headers = {"Authorization": f"token {token}"} if token else {}

    def get_default_branch(owner: str, repo: str) -> str:
        """
        Get the default branch name of a GitHub repository.

        This helper function queries the GitHub API to retrieve metadata
        about the repository and returns the default branch (usually ``main`` or ``master``).

        Parameters
        ----------
        owner : str
            GitHub username or organization name.
        repo : str
            Name of the GitHub repository.

        Returns
        -------
        str
            The name of the default branch, such as ``main`` or ``master``.

        Raises
        ------
        requests.exceptions.RequestException
            If the API request to GitHub fails.

        Notes
        -----
        - Uses the global ``headers`` variable for authentication if a token is provided.
        - Falls back to ``main`` if the default branch cannot be determined.

        Examples
        --------
        >>> get_default_branch("dest-ash", "bnn_for_14C_calibration")
        'main'
        """
        repo_info_url = f"https://api.github.com/repos/{owner}/{repo}"
        r_info = requests.get(repo_info_url, headers=headers, timeout=timeout)
        r_info.raise_for_status()
        return r_info.json().get("default_branch", "main")

    def _download_folder(api_url: str, local_dir: Path) -> None:
        """
        Recursively download the contents of a GitHub folder and apply mappings
        defined in a local or remote ``drive_map.json`` file.

        This internal function is called by :func:`download_github_with_drive_map`
        and is responsible for traversing directories, downloading files,
        and handling external resources (Google Drive / Hugging Face) based on mappings.

        Parameters
        ----------
        api_url : str
            GitHub API URL for the folder to download.
        local_dir : Path
            Local directory where the contents will be stored.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the API URL does not contain valid repository information.
        RuntimeError
            If the download from all mapped sources fails.

        Notes
        -----
        - This function creates directories recursively.
        - It will skip files or folders already handled by ``drive_map.json``.
        - Partial caches are cleaned if unrecoverable errors occur.

        Examples
        --------
        >>> _download_folder(
        ...     "https://api.github.com/repos/user/repo/contents/models",
        ...     Path("./models")
        ... )
        📄 Found drive_map.json in https://api.github.com/repos/user/repo/contents/models, downloading → ./models/drive_map.json
        """
        local_dir.mkdir(parents=True, exist_ok=True)

        # Récupère la liste des fichiers du dossier GitHub
        response = requests.get(api_url, headers=headers, timeout=timeout)
        response.raise_for_status()
        items = response.json()

        # Charge drive_map.json depuis GitHub si présent
        drive_map: Dict[str, Any] = {}
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
            owner, repo = parts[i + 1], parts[i + 2]
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
                        download_from_huggingface(
                            mapped["huggingface"],
                            local_path,
                            timeout=timeout,
                            sleep_time=sleep_time
                        )
                        success = True
                    except Exception as e_hf:
                        print(f"⚠️ Hugging Face download failed for {relative_name}: {e_hf}")

                # Google Drive fallback
                if not success and "drive" in mapped:
                    try:
                        download_from_google_drive(
                            mapped["drive"],
                            local_path,
                            sleep_time=sleep_time
                        )
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
                    raise RuntimeError(
                        f"❌ Failed to download {relative_name} from both Hugging Face and Google Drive"
                    )

                # ✅ Already downloaded via external sources, skipping the GitHub downloading step
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




def download_cache_lib_data(overwrite: bool = False) -> None:
    """
    Download and initialize the local cache directory for the library.

    This function manages the creation (or re-creation) of the local cache directory
    used by the package to store pre-downloaded model and data files.  
    If ``overwrite`` is set to True, any existing cache will be removed before re-downloading.

    Parameters
    ----------
    overwrite : bool, optional
        Whether to force re-downloading and replace any existing cache directory (default is False).

    Returns
    -------
    None

    Notes
    -----
    - The cache directory ``.bnn_for_14C_calibration`` is created in the user's home directory.
    - Requires an active internet connection for the first run.
    - Disk space usage remains below 1 GB.
    - This function internally calls the functions `clear_cache` and `download_github_with_drive_map`.

    Examples
    --------
    >>> download_cache_lib_data(overwrite=False)
    This may be the first time you need package functions that use 
    local cache data to work. A local cache directory is going to be 
    created and will be used after if needed without a new downloading
    unless you delete the cache. To download this cache, network connection 
    must be available; also some disk space is required (less than 1 GB total).
    ******************** Creating cache directory at: /home/user/.bnn_for_14C_calibration ********************
    ✅ Cache directory created at: /home/user/.bnn_for_14C_calibration, and filled with all the 
    contents of the 'models' directory downloaded from GitHub and Google Drive.

    >>> download_cache_lib_data(overwrite=True)
    overwrite is True : the cache will be cleared before 
    downloading it again...
    🗑️ removing cache directory at : /home/user/.bnn_for_14C_calibration...
    🗑️ cache removed!
    ******************** Creating cache directory at: /home/user/.bnn_for_14C_calibration ********************
    ✅ Cache directory created at: /home/user/.bnn_for_14C_calibration, and filled with all the 
    contents of the 'models' directory downloaded from GitHub and Google Drive.
    """
    # Vérifie si le cache doit être téléchargé ou régénéré
    if overwrite or not (CACHE_DIR.exists() and CACHE_DIR.is_dir()):
        if overwrite:
            print(f"""
                overwrite is {overwrite} : the cache will be cleared before 
                downloading it again...
            """)
            clear_cache()
        else:
            print("""
                This may be the first time you need package functions that use 
                local cache data to work. A local cache directory is going to be 
                created and will be used after if needed without a new downloading
                unless you delete the cache. To download this cache, network connection 
                must be available; also some disk space is required (less than 1 GB total).
            """)

        print(f"******************** Creating cache directory at: {CACHE_DIR} ********************")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR_LOCAL.mkdir(exist_ok=True)

        # Télécharge les données de modèles depuis GitHub (et éventuellement Google Drive)
        download_github_with_drive_map(MODELS_DIR_API_URL, MODELS_DIR_LOCAL)

        print(f"""
        ✅ Cache directory created at: {CACHE_DIR}, and filled with all the 
        contents of the 'models' directory downloaded from GitHub and Google Drive.
        """)
    else:
        print(f"""
            An existing cache directory is located at {CACHE_DIR} and overwrite is {overwrite}.
            If you wish to force the cache download, set overwrite to True.
        """)



# fonctions publiques du module
__all__ = [
    # fonctions de gestion du cache local
    "download_cache_lib_data",
    "clear_cache"
]


# toutes les fonctions du module
all_functions = [
    "clear_cache",
    "is_google_drive_url",
    "extract_drive_file_id",
    "download_from_google_drive",
    "download_from_huggingface",
    "download_github_with_drive_map",
    "download_cache_lib_data"

]

if __name__ == "__main__":
    download_cache_lib_data()

