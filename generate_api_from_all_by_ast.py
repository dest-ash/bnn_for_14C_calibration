# utilise ast pour générer les fichiers .md dans "docs/reference" à partir des variables __all__ de différents modules
# ces fichiers .md sont utilisés pour mkdocs pour générer la documentation à partir des docstrings de fonctions python

import ast, pathlib

SRC = pathlib.Path("src")
PKG_DIR = SRC / "bnn_for_14C_calibration"
OUT_DIR = pathlib.Path("docs/reference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in PKG_DIR.glob("*.py"):
    if p.name == "__init__.py":
        continue
    modname = f"bnn_module.{p.stem}"
    src = p.read_text()
    tree = ast.parse(src)

    all_names = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "__all__":
                    # try to evaluate simple literal list/tuple
                    try:
                        value = ast.literal_eval(node.value)
                        all_names = [str(x) for x in value]
                    except Exception:
                        all_names = None
                    break
        if all_names is not None:
            break

    if all_names is None:
        # fallback: collect defined functions/classes with non-underscore names
        defined = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
                defined.append(node.name)
        names = defined
    else:
        names = all_names

    if not names:
        print(f"Skipping {modname}: no public names")
        continue

    md = f"# {p.stem}\n\n::: {modname}\n    selection:\n      members:\n"
    for n in names:
        md += f"        - {n}\n"
    md += "    options:\n      show_root_heading: false\n      show_root_toc_entry: false\n"

    out_file = OUT_DIR / f"{p.stem}.md"
    out_file.write_text(md)
    print(f"Wrote {out_file} (members: {len(names)})")
