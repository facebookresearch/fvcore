#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

for PV in 3.6 3.7 3.8
do
   PYTHON_VERSION=$PV bash packaging/build_conda.sh
done

ls -Rl packaging

for version in 36 37 38
do
    (cd packaging/out && conda convert -p win-64 linux-64/fvcore-*-py$version.tar.bz2)
    (cd packaging/out && conda convert -p osx-64 linux-64/fvcore-*-py$version.tar.bz2)
done

ls -Rl packaging

for dir in win-64 osx-64 linux-64
do
    this_out_dir=packaging/output_files/$dir
    mkdir -p $this_out_dir
    cp packaging/out/$dir/*.tar.bz2 $this_out_dir
done

ls -Rl packaging
