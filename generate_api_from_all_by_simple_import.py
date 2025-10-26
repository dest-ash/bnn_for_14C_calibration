# utilise un simple import pour générer les fichiers .md dans "docs/reference" à partir des variables __all__ de différents modules
# ces fichiers .md sont utilisés pour mkdocs pour générer la documentation à partir des docstrings de fonctions python

import importlib, inspect, sys, pathlib

SRC = "src"                               # chemin vers le code source
PKG = "bnn_for_14C_calibration"           # package racine
OUT_DIR = pathlib.Path("docs/reference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, SRC)

# Découvre les modules à traiter automatiquement (ici : tous les .py du package)
pkg_path = pathlib.Path(SRC) / PKG
modules = []
for p in pkg_path.glob("*.py"):
    if p.name == "__init__.py":
        continue
    modules.append(f"{PKG}.{p.stem}")

# ajoute le package racine si tu veux documenter __init__ aussi
# modules.insert(0, PKG)

for modname in modules:
    try:
        m = importlib.import_module(modname)
    except Exception as e:
        print(f"WARNING: could not import {modname}: {e}")
        continue

    names = getattr(m, "__all__", None)
    if not names:
        # fallback: fonctions/classes définies *dans* le module et non privées
        names = [n for n, o in inspect.getmembers(m)
                 if (inspect.isfunction(o) or inspect.isclass(o)) and o.__module__ == m.__name__ and not n.startswith("_")]

    if not names:
        print(f"NOTE: no public names for {modname}, skipping")
        continue

    md = f"# {modname.split('.')[-1]}\n\n::: {modname}\n    selection:\n      members:\n"
    for n in names:
        md += f"        - {n}\n"
    md += "    options:\n      show_root_heading: false\n      show_root_toc_entry: false\n"

    out_file = OUT_DIR / f"{modname.split('.')[-1]}.md"
    out_file.write_text(md)
    print(f"Wrote {out_file} (members: {len(names)})")
