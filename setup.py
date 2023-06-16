#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
from os import path

from setuptools import find_packages, setup


def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "fvcore", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # Used by CI to build nightly packages. Users should never use it.
    # To build a nightly wheel, run:
    # BUILD_NIGHTLY=1 python setup.py sdist
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%Y%m%d")
        # pip can perform proper comparison for ".post" suffix,
        # i.e., "1.1.post1234" >= "1.1"
        version = version + ".post" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


setup(
    name="fvcore",
    version=get_version(),
    author="FAIR",
    license="Apache 2.0",
    url="https://github.com/facebookresearch/fvcore",
    description="Collection of common code shared among different research "
    "projects in FAIR computer vision team",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "tqdm",
        "termcolor>=1.1",
        "Pillow",
        "tabulate",
        "iopath>=0.1.7",
        "dataclasses; python_version<'3.12'",
    ],
    extras_require={"all": ["shapely"]},
    packages=find_packages(exclude=("tests",)),
)
