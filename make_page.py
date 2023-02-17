#!/usr/bin/env python3
from shutil import copytree, rmtree, copy
from pathlib import Path
from sphinx.ext.apidoc import main as sphinx_apidoc
from sphinx.cmd.build import main as sphinx_build

# Regen the API docs
sphinx_apidoc(
    [
        "--ext-autodoc",
        "--ext-mathjax",
        "-o",
        "docs_src/ref",
        "hylight/",
        "hylight/cli/*",
    ]
)
Path("docs_src/ref/modules.rst").unlink()

# Build the static content
sphinx_build(["-M", "html", "./docs_src", "./public"])


src = Path("public/html/")
docs = Path("docs")

# rm -r docs/*
for elem in docs.glob("*"):
    if elem.is_dir():
        rmtree(elem)
    else:
        elem.unlink()

# cp -r public/html/* docs
for elem in src.glob("*"):
    if elem.is_dir():
        copytree(elem, docs / elem.relative_to(src))
    else:
        copy(elem, docs)

# Ensure the presence of the .nojekyll for GitHub
(docs / ".nojekyll").touch()
