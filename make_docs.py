#!/usr/bin/env python3
"""Generate the documentation for Hylight.

Usage:
    hatch shell
    ./make_docs.py
    exit
"""
from shutil import copytree, rmtree, copy
from pathlib import Path
from sphinx.ext.apidoc import main as sphinx_apidoc
from sphinx.cmd.build import main as sphinx_build

# Regen the API docs
errcode = sphinx_apidoc(
    [
        "--ext-autodoc",
        "--ext-mathjax",
        "-M",
        "-t",
        "docs_src/_templates/",
        "-o",
        "docs_src/ref",
        "hylight/",
        "hylight/cli/*",
    ]
)
if errcode != 0:
    exit(errcode)

Path("docs_src/ref/modules.rst").unlink()

# Build the static content
errcode = sphinx_build(["-M", "html", "./docs_src", "./public"])
if errcode != 0:
    exit(errcode)

errcode = sphinx_build(["-M", "latexpdf", "./docs_src", "./public"])
if errcode != 0:
    exit(errcode)


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
