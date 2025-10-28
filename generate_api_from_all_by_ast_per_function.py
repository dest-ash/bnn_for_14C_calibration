#!/usr/bin/env python3
"""
Génère docs/reference/<module>.md pour chaque module du package en analysant AST pour lire __all__ (statique).
Ajoute une phrase de description après le titre du module, prise dans MODULE_DESCRIPTIONS ou dans module_descriptions.json.

Ces fichiers .md sont utilisés pour mkdocs pour générer la documentation à partir des docstrings de fonctions python
"""

import ast
import pathlib
import json

# CONFIG -------------------------------------------------------
SRC_DIR = pathlib.Path("src")
PKG = "bnn_for_14C_calibration"
PKG_DIR = SRC_DIR / PKG
OUT_DIR = pathlib.Path("docs/reference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DESCRIPTIONS_FILE = pathlib.Path("module_descriptions.json")

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


# options to write in each block (indentation EXACT)
DEFAULT_OPTIONS = [
    "    options:",
    "      show_root_heading: true",
    "      show_root_toc_entry: false",
    "      show_if_no_docstring: true",
    "      show_signature: true",
    "      separate_signature: false",
    "      show_source: true",
    "      resolve_aliases: false",
    "      show_submodules: false",
]
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

def extract_all_and_defined_names(py_path):
    """Return (all_names_or_None, defined_public_names_list) from AST analysis."""
    src = py_path.read_text(encoding="utf8")
    tree = ast.parse(src)
    all_names = None
    defined_names = []
    for node in tree.body:
        # detect __all__ = [...] literal (list or tuple)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "__all__":
                    try:
                        value = ast.literal_eval(node.value)
                        if isinstance(value, (list, tuple)):
                            all_names = [str(x) for x in value]
                    except Exception:
                        all_names = None
                    break
        # collect function and class names defined in this module
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                defined_names.append(node.name)
    return all_names, defined_names

def main():
    descriptions = load_descriptions()

    if not PKG_DIR.exists():
        print(f"ERROR: package dir {PKG_DIR} does not exist.")
        return

    for p in sorted(PKG_DIR.glob("*.py")):
        if p.name == "__init__.py":
            continue
        modname = f"{PKG}.{p.stem}"
        all_names, defined_public = extract_all_and_defined_names(p)
        if all_names is not None:
            names = all_names
        else:
            names = defined_public

        if not names:
            print(f"Skipping {modname}: no public names found")
            continue

        desc = descriptions.get(modname) or descriptions.get(p.stem) or ""

        shortname = p.stem
        md_lines = []
        md_lines.append(f"# {shortname}\n")
        if desc:
            md_lines.append(f"{desc}\n")
        for n in names:
            md_lines.append(f"::: {modname}.{n}\n")
            # md_lines.append("    options:\n")
            # md_lines.append("      show_root_heading: false\n")
            # md_lines.append("      show_root_toc_entry: false\n")
            # md_lines.append("      show_if_no_docstring: true\n")
            for line in DEFAULT_OPTIONS:
                md_lines.append(f"{line}\n")

        out_file = OUT_DIR / f"{shortname}.md"
        out_file.write_text("".join(md_lines), encoding="utf8")
        print(f"Wrote {out_file} (added functions to the doc: {len(names)})")

if __name__ == "__main__":
    main()
