#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment Checker for AI/ML Data Preprocessing Notebooks
--------------------------------------------------------
Verifies that all Python packages listed in requirements.txt
are installed and importable.

Handles:
    - Standard version format: package>=1.0
    - pip freeze URL format: package @ file:///path
    - PyPI-to-import name mapping
    - Special cases: spaCy models, google.protobuf, PyQt5.sip, etc.
    - Skip optional/Windows-only packages
"""

import importlib
import sys
from pathlib import Path
from importlib.metadata import version as get_version

REQUIREMENTS_FILE = Path(__file__).parent.parent / "requirements.txt"


def parse_requirements(filepath):
    """
    Parse requirements.txt and extract importable package names.

    Handles:
        - "numpy>=1.21.0" -> "numpy"
        - "package @ file:///path" -> "package"
        - "package @ https://..." -> "package"

    Args:
        filepath (Path): Path to requirements.txt.

    Returns:
        list: List of importable package names.
    """
    if not filepath.exists():
        print(f"Error: Requirements file not found at {filepath}")
        sys.exit(1)

    packages = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Extract base package name
            if " @ " in line:
                pkg = line.split(" @ ")[0].strip()
            elif ">=" in line:
                pkg = line.split(">=")[0].strip()
            elif "==" in line:
                pkg = line.split("==")[0].strip()
            elif "<" in line:
                pkg = line.split("<")[0].strip()
            else:
                pkg = line.strip()

            # Map to import name
            pkg = map_to_import_name(pkg)

            if pkg and pkg not in packages:
                packages.append(pkg)

    return packages


def map_to_import_name(pkg):
    """
    Map PyPI package names to their importable module names.

    Args:
        pkg (str): Package name from requirements.txt.

    Returns:
        str or None: Importable module name, or None if should be skipped.
    """
    # Skip Windows-only packages on non-Windows
    if pkg in ("pywin32", "win_inet_pton"):
        if sys.platform != "win32":
            return None
        if pkg == "pywin32":
            return "win32api"  # actual import name

    # Skip optional CLI packages not needed for notebooks
    if pkg in ("typer_slim", "ruff", "black"):
        return None

    # Standard mapping
    mapping = {
        # Image processing
        "opencv-python": "cv2",
        "opencv-python-headless": "cv2",
        "pillow": "PIL",
        "scikit-image": "skimage",

        # Machine learning
        "scikit-learn": "sklearn",
        "imbalanced-learn": "imblearn",
        "statsmodels": "statsmodels",

        # NLP
        "nltk": "nltk",
        "spacy": "spacy",
        "es_core_news_sm": "es_core_news_sm",

        # Visualisation
        "wordcloud": "wordcloud",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",

        # Audio
        "librosa": "librosa",
        "soundfile": "soundfile",
        "soxr": "soxr",

        # Data
        "pandas": "pandas",
        "numpy": "numpy",
        "scipy": "scipy",

        # Jupyter
        "ipykernel": "ipykernel",
        "ipython": "IPython",
        "jupyter_client": "jupyter_client",
        "jupyter_core": "jupyter_core",
        "typer-slim": "typer",

        # Web / API
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "starlette": "starlette",
        "gradio": "gradio",
        "gradio_client": "gradio_client",
        "requests": "requests",
        "httpx": "httpx",
        "httpcore": "httpcore",
        "websockets": "websockets",

        # Utilities
        "click": "click",
        "rich": "rich",
        "typer": "typer",
        "python-dotenv": "dotenv",
        "Faker": "faker",
        "kagglehub": "kagglehub",
        "tqdm": "tqdm",
        "joblib": "joblib",
        "threadpoolctl": "threadpoolctl",
        "packaging": "packaging",

        # Config / serialisation
        "PyYAML": "yaml",
        "toml": "toml",
        "tomli": "tomli",
        "tomlkit": "tomlkit",
        "orjson": "orjson",
        "msgpack": "msgpack",

        # Crypto / security
        "cryptography": "cryptography",
        "pycparser": "pycparser",
        "cffi": "cffi",
        "certifi": "certifi",
        "charset_normalizer": "charset_normalizer",
        "idna": "idna",
        "urllib3": "urllib3",

        # DNS / networking
        "dnspython": "dns",

        # Fonts
        "fonttools": "fontTools",

        # Templating
        "Jinja2": "jinja2",
        "MarkupSafe": "markupsafe",

        # Markdown
        "markdown-it-py": "markdown_it",

        # Protocol Buffers
        "protobuf": "google.protobuf",

        # SOCKS
        "PySocks": "socks",

        # ZeroMQ
        "pyzmq": "zmq",

        # Windows
        "win_inet_pton": "win_inet_pton",

        # Other
        "defusedxml": "defusedxml",
        "lxml": "lxml",
        "beautifulsoup4": "bs4",
        "soupsieve": "soupsieve",

        # PyQt
        "PyQt5": "PyQt5",
        "PyQt5_sip": "PyQt5.sip",

        # Misc
        "munkres": "munkres",
        "patsy": "patsy",
        "pydantic": "pydantic",
        "pydantic_core": "pydantic_core",
        "pydantic_extra_types": "pydantic_extra_types",
        "pydantic_settings": "pydantic_settings",

        # Fix case-sensitive packages
        "Brotli": "brotli",
        "Pygments": "pygments",
        "python-dateutil": "dateutil",
    }

    return mapping.get(pkg, pkg.replace("-", "_"))


def check_package(pkg_name):
    """
    Attempt to import a package and get its version.

    Special handling for:
        - spaCy models (es_core_news_sm)
        - PyQt5.sip
        - google.protobuf
        - dotenv (python-dotenv)

    Args:
        pkg_name (str): Importable module name.

    Returns:
        tuple: (bool, str) – (installed, version)
    """
    # Special: spaCy model
    if pkg_name == "es_core_news_sm":
        try:
            import spacy
            nlp = spacy.load("es_core_news_sm")
            return True, nlp.meta.get("version", "unknown")
        except (ImportError, OSError):
            return False, "not installed"

    # Special: PyQt5.sip
    if pkg_name == "PyQt5.sip":
        try:
            import PyQt5.sip as sip
            return True, sip.SIP_VERSION_STR
        except ImportError:
            return False, "not installed"

    # Special: google.protobuf
    if pkg_name == "google.protobuf":
        try:
            import google.protobuf
            return True, google.protobuf.__version__
        except ImportError:
            return False, "not installed"

    # Special: dotenv (python-dotenv)
    if pkg_name == "dotenv":
        try:
            import dotenv
            return True, getattr(dotenv, "__version__", "unknown")
        except ImportError:
            return False, "not installed"

    # General case
    try:
        module = importlib.import_module(pkg_name)
        # Get version using importlib.metadata for standard packages
        try:
            # Use the package name from mapping (original PyPI name)
            # We don't have the original name here; fallback to __version__
            version = getattr(module, "__version__", None)
            if version is None:
                version = getattr(module, "version", "unknown")
        except Exception:
            version = "unknown"
        return True, version
    except ImportError:
        return False, "not installed"


def main():
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)
    print(f"Python version: {sys.version}")
    print(f"Requirements file: {REQUIREMENTS_FILE}")
    print()

    packages = parse_requirements(REQUIREMENTS_FILE)
    print(f"Found {len(packages)} packages to check.")
    print()

    all_ok = True
    for pkg in packages:
        ok, version = check_package(pkg)
        status = "[OK]" if ok else "[FAIL]"
        if len(str(version)) > 40:
            version = str(version)[:40] + "..."
        print(f"{status} {pkg:<30} {version}")
        if not ok:
            all_ok = False

    print("\n" + "=" * 70)
    if all_ok:
        print("[OK] All packages are installed. Environment is ready.")
    else:
        print("[FAIL] Some packages are missing.")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()