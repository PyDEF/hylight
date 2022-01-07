import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hylight",
    version="0.1.0",
    author="Théo Cavignac",
    author_email="theo.cavignac@gmail.com",
    description="Hylight is post-processing package for luminescence related ab initio computations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PyDEF/hylight",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    keywords=[
        "chemistry",
        "computational chemistry",
    ],
)
