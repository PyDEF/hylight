# About

This document is addressed to the members of the PyDEF team.
If you are not part of this team and are interested in contributing, please
contact Camille Latouche (camille.latouche at cnrs-imn.fr).

# Tooling

This project uses hatch for the build system and sphinx to generate the documentation.
To set up the development environment, first install hatch (`pip install hatch`).
Then run `hatch shell` to start the development environment.
Finally, install all dependencies (including the development ones) with `pip install -e .[dev,plotting,phonopy,hdf5]`.

The formatting of the code must follow [black](https://black.readthedocs.io/en/stable/) rules.
Apply `black .` at the root of the project.
I also recommend that you regularly check the code for common errors with
`flake8 hylight` or `ruff hylight`, as well as type errors with `mypy hylight`.

# TODO list for release

- [ ] Check that all the features are properly documented with docstrings (using flake8 with the [docstrings plugin](https://pypi.org/project/flake8-docstrings/) helps).
- [ ] Apply [black](https://black.readthedocs.io/en/stable/) to the project: `black hylight/`
- [ ] Bump the version in `hylight/__init__.py`
- [ ] Add a new section at the **top** of `CHANGELOG.md` (versions should decrease in the file) that sum up the new features and bug fixes.
- [ ] Update the documentation
```sh
hatch shell
pip install .[dev,plotting,phonopy,hdf5]
cd docs_src && make regen && cd ..  # only if there is new/renamed/removed modules
python ./make_docs.py
exit
```
- [ ] Commit the documentation and code
- [ ] Tag the commit with the name of the version `git tag v1.0.0`
- [ ] Push the commits to GitHub `git push github main`
- [ ] Push the tags to GitHub `git push github --tags`
- [ ] Make sure the documentation is properly displayed on the page `http://pydef.github.io/hylight`
- [ ] Publish the release to PyPI (see the private wiki of the project).
```sh
hatch build
hatch publish
```
