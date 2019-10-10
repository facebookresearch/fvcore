#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import glob
import importlib
from os.path import basename, dirname, isfile, join, sys

if __name__ == "__main__":
    if len(sys.argv) > 1:  # pyre-ignore[16]
        # Parse from flags.
        module_names = [
            n for n in sys.argv if n.startswith("bm_")
        ]  # pyre-ignore[16]
    else:
        # Get all the benchmark files (starting with "bm_").
        bm_files = glob.glob(join(dirname(__file__), "bm_*.py"))
        module_names = [
            basename(f)[:-3]
            for f in bm_files
            if isfile(f) and not f.endswith("bm_main.py")
        ]

    for module_name in module_names:
        module = importlib.import_module(module_name)
        for attr in dir(module):
            # Run all the functions with names "bm_*" in the module.
            if attr.startswith("bm_"):
                print(
                    "Running benchmarks for " + module_name + "/" + attr + "..."
                )
                getattr(module, attr)()
