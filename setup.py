#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import, print_function

from os import path
from io import open

from setuptools import find_packages, setup

DISTNAME = "samplics"
DESCRIPTION = "Select, weight and analyze complex sample data"
AUTHOR = "Mamadou S Diallo"
AUTHOR_EMAIL = "msdiallo@QuantifyAfrica.org"
URL = "https://github.com/MamadouSDiallo/samplics"
REQUIRES_PYTHON = ">=3.6"
REQUIRED = [
    "scipy>=1.4.0",
    "numpy>=1.15.0",
    "pandas>=0.25.0",
    "statsmodels>=0.10.0",
]
DEV_PACKAGES = ["pytest>=5.3.2", "black>=19.10", "mypy>=0.761"]
VERSION = "0.0.4"

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
try:
    with open(path.join(here, "README.rst"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(path.join(here, DISTNAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

setup(
    name=DISTNAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    # long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
        "Operating System :: OS Independent",
        "Intended Audience :: Survey Analysts",
        "Intended Audience :: Statisticians",
        "Intended Audience :: Data Scientists",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineeing :: Statistics",
    ],
    keywords=["sampling", "sample", "weighting", "estimation", "survey"],
    packages=find_packages(where="src", exclude=["datasets", "docs", "tests"]),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=REQUIRED,
    extras_require={"dev": ["pytest>=5.3.2", "black>=19.10", "mypy>=0.761"]},
    test_suite="tests",
)
