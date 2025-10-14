import subprocess
import sys
from pathlib import Path

# --- Utilitaire simple pour exécuter une commande shell pour automatiser la publication de la librairie et gérer la version ---
def run(cmd, dry=False):
    print(f"👉 {cmd}")
    if not dry:
        subprocess.run(cmd, shell=True, check=True)

def ask_confirmation(prompt):
    """Demande une confirmation utilisateur."""
    reply = input(f"{prompt} [y/N]: ").strip().lower()
    return reply == "y"

def main(bump_type="patch", dry=False, test=False):
    print("🚀 Secure Release Script for ma_librairie\n")

    # 1️⃣ Nettoyage
    if ask_confirmation("🧹 Clean previous builds?"):
        run("rm -rf dist/ build/ *.egg-info", dry)

    # 2️⃣ Build
    if ask_confirmation("📦 Build package?"):
        run("python -m build", dry)

    # 3️⃣ Bump version
    if ask_confirmation(f"🔢 Bump version ({bump_type})?"):
        run(f"bumpver update --{bump_type}", dry)

    # 4️⃣ Push Git
    if ask_confirmation("🚀 Push commit and tag to GitHub?"):
        run("git push && git push --tags", dry)

    # 5️⃣ Création Release GitHub
    tag = subprocess.check_output("git describe --tags --abbrev=0", shell=True).decode().strip()
    if ask_confirmation(f"🏷️ Create GitHub release for {tag}?"):
        run(f'gh release create {tag} dist/* --title "Release {tag}" --notes "Automated release."', dry)

    # 6️⃣ Upload TestPyPI ou PyPI
    repo_url = "https://test.pypi.org/legacy/" if test else "https://upload.pypi.org/legacy/"
    repo_name = "TestPyPI" if test else "PyPI"
    if ask_confirmation(f"☁️ Upload to {repo_name}?"):
        run(f"twine upload --repository-url {repo_url} dist/*", dry)

    print("\n✅ Done!")


if __name__ == "__main__":
    bump_type = sys.argv[1] if len(sys.argv) > 1 else "patch"
    dry = "--dry" in sys.argv
    test = "--test" in sys.argv
    main(bump_type=bump_type, dry=dry, test=test)
