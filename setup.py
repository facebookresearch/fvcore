#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="fvcore",
    version="0.1",
    author="FAIR",
    url="https://github.com/facebookresearch/fvcore",
    description="Collection of common code shared among different research "
        "projects in FAIR computer vision team",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "tqdm",
        "portalocker",
        "termcolor>=1.1",
        "shapely",
    ],
    packages=find_packages(exclude=("tests")),
)
