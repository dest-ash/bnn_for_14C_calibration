#!/usr/bin/env python3

"""
Génère docs/reference/<module>.md pour chaque module du package en lisant __all__ (import-based).
Ajoute une phrase de description après le titre du module, prise dans MODULE_DESCRIPTIONS ou dans module_descriptions.json.

Ces fichiers .md sont utilisés pour mkdocs pour générer la documentation à partir des docstrings de fonctions python
"""

import importlib
import inspect
import sys
import pathlib
import json

# CONFIG -------------------------------------------------------
SRC = "src"                               # chemin vers le code source
PKG = "bnn_for_14C_calibration"           # package racine
OUT_DIR = pathlib.Path("docs/reference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Optional: load descriptions from JSON file (module_descriptions.json)
DESCRIPTIONS_FILE = pathlib.Path("module_descriptions.json")

# Fallback: inline dictionary (module name -> description string)
MODULE_DESCRIPTIONS = {
    # courte description des modules
    "bnn_models_built_in_utils": "Description of the utilities functions for Bayesian modeling: ",
    "bnn_models_built_in": "Description of built in functions for Baysian modelling: ",
    "calib_plot_functions": "Description of plotting functions: ",
    "calibration_utils": "Description of helpers functions for calibration step: ",
    "calibration": "Description of the calibration functions implemented in the Python library `bnn_for_14C_calibration`: ",
    "manage_cache": "Description of the functions implemented to store and manage lib data in a local cache on the disk: ",
    "utils": "Description of general utilities functions: ",
}
# --------------------------------------------------------------

def load_descriptions():
    if DESCRIPTIONS_FILE.exists():
        try:
            data = json.loads(DESCRIPTIONS_FILE.read_text(encoding="utf8"))
            if isinstance(data, dict):
                return data
            else:
                print(f"Warning: {DESCRIPTIONS_FILE} exists but does not contain a JSON object. Using inline dict.")
        except Exception as e:
            print(f"Warning: failed to read {DESCRIPTIONS_FILE}: {e}. Using inline dict.")
    return MODULE_DESCRIPTIONS

def main():
    sys.path.insert(0, SRC)
    descriptions = load_descriptions()

    pkg_path = pathlib.Path(SRC) / PKG
    if not pkg_path.exists():
        print(f"ERROR: package path {pkg_path} does not exist.")
        return

    # Découvre automatiquement les modules .py sous le package (skip __init__.py)
    modules = []
    for p in sorted(pkg_path.glob("*.py")):
        if p.name == "__init__.py":
            continue
        modules.append(f"{PKG}.{p.stem}")

    # Optionnel : include package __init__ (décommente si souhaité)
    # modules.insert(0, PKG)

    for modname in modules:
        try:
            m = importlib.import_module(modname)
        except Exception as e:
            print(f"WARNING: could not import {modname}: {e}")
            continue

        # Récupère __all__ si présent, sinon fallback aux définitions locales publiques
        names = getattr(m, "__all__", None)
        if not names:
            names = [n for n, o in inspect.getmembers(m)
                     if (inspect.isfunction(o) or inspect.isclass(o))
                     and o.__module__ == m.__name__ and not n.startswith("_")]

        if not names:
            print(f"NOTE: no public names for {modname}, skipping")
            continue

        # prepare description: prefer full module name key, else last part (module shortname)
        desc = descriptions.get(modname) or descriptions.get(modname.split(".")[-1]) or ""

        # build markdown content
        shortname = modname.split(".")[-1]
        md_lines = []
        md_lines.append(f"# {shortname}\n")
        if desc:
            md_lines.append(f"{desc}\n")
        md_lines.append(f"::: {modname}\n")
        md_lines.append("    selection:\n")
        md_lines.append("      members:\n")
        for n in names:
            md_lines.append(f"        - {n}\n")
        md_lines.append("    options:\n")
        md_lines.append("      show_root_heading: false\n")
        md_lines.append("      show_root_toc_entry: false\n")
        md_lines.append("      show_if_no_docstring: true\n")

        out_file = OUT_DIR / f"{shortname}.md"
        out_file.write_text("".join(md_lines), encoding="utf8")
        print(f"Wrote {out_file} (members: {len(names)})")

if __name__ == "__main__":
    main()
