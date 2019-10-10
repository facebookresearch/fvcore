#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup

setup(
    name="fvcore",
    version="0.1",
    author="FAIR",
    url="unknown",
    description="Bags of Reusable Components for Computer Vision Research",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "tqdm",
        "portalocker",
        "termcolor>=1.1",
        "shapely",
    ],
    packages=find_packages(exclude=("configs", "tests")),
)
