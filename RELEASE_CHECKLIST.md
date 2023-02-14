# TODO list for release

- [ ] Check that all the features are properly documented with docstrings.
- [ ] Bump the version in `hylight/__init__.py`
- [ ] Apply [black](https://black.readthedocs.io/en/stable/) to the project: `black hylight/`
- [ ] Add a new section at the **top** of `CHANGELOG.md` (versions should decrease in the file) that sum up the new features and bug fixes.
- [ ] Update the documentation
```sh
cd docs
make wipe
make html
make latexpdf
cd ..
```
- [ ] Commit the documentation and code
- [ ] Tag the commit with the name of the version `git tag v1.0.0`
- [ ] Push the commits to GitHub `git push github main`
- [ ] Push the tags to GitHub `git push github v1.0.0`
- [ ] Make sure the documentation is properly displayed on the page `http://hylight.github.io/`
- [ ] Publish the release to PyPI (see the private wiki of the project).
