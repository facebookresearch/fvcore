#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict

import glob
import importlib
import sys
from os.path import basename, dirname, isfile, join


def main() -> None:
    if len(sys.argv) > 1:
        # Parse from flags.
        module_names = [n for n in sys.argv if n.startswith("bm_")]
    else:
        # Get all the benchmark files (starting with "bm_").
        bm_files = glob.glob(join(dirname(__file__), "bm_*.py"))
        module_names = [
            basename(f)[:-3]
            for f in bm_files
            if isfile(f) and not f.endswith("bm_main.py")
        ]

    # pyre-fixme[16]: `List` has no attribute `__iter__`.
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for attr in dir(module):
            # Run all the functions with names "bm_*" in the module.
            if attr.startswith("bm_"):
                # pyre-fixme[16]: `str` has no attribute `__add__`.
                # pyre-fixme[58]: `+` is not supported for operand types `str` and
                #  `Any`.
                print("Running benchmarks for " + module_name + "/" + attr + "...")
                getattr(module, attr)()


# pyre-fixme[16]: `str` has no attribute `__eq__`.
if __name__ == "__main__":
    main()  # pragma: no cover
