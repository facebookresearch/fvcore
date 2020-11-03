# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
from typing import IO, Any, Callable, Dict, List, Union

import yaml
from fvcore.common.file_io import PathManager
from yacs.config import CfgNode as _CfgNode


BASE_KEY = "_BASE_"


class CfgNode(_CfgNode):
    """
    Our own extended version of :class:`yacs.config.CfgNode`.
    It contains the following extra features:

    1. The :meth:`merge_from_file` method supports the "_BASE_" key,
       which allows the new CfgNode to inherit all the attributes from the
       base configuration file.
    2. Keys that start with "COMPUTED_" are treated as insertion-only
       "computed" attributes. They can be inserted regardless of whether
       the CfgNode is frozen or not.
    3. With "allow_unsafe=True", it supports pyyaml tags that evaluate
       expressions in config. See examples in
       https://pyyaml.org/wiki/PyYAMLDocumentation#yaml-tags-and-python-types
       Note that this may lead to arbitrary code execution: you must not
       load a config file from untrusted sources before manually inspecting
       the content of the file.
    """

    @classmethod
    def _open_cfg(cls, filename: str) -> Union[IO[str], IO[bytes]]:
        """
        Defines how a config file is opened. May be overridden to support
        different file schemas.
        """
        return PathManager.open(filename, "r")

    @classmethod
    def load_yaml_with_base(cls, filename: str, allow_unsafe: bool = False) -> None:
        """
        Just like `yaml.load(open(filename))`, but inherit attributes from its
            `_BASE_`.

        Args:
            filename (str or file-like object): the file name or file of the current config.
                Will be used to find the base config file.
            allow_unsafe (bool): whether to allow loading the config file with
                `yaml.unsafe_load`.

        Returns:
            (dict): the loaded yaml
        """
        with cls._open_cfg(filename) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Loading config {} with yaml.unsafe_load. Your machine may "
                    "be at risk if the file contains malicious content.".format(
                        filename
                    )
                )
                f.close()
                with cls._open_cfg(filename) as f:
                    cfg = yaml.unsafe_load(f)  # pyre-ignore

        # pyre-ignore
        def merge_a_into_b(a: Dict[Any, Any], b: Dict[Any, Any]) -> None:
            # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(
                        b[k], dict
                    ), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        if BASE_KEY in cfg:
            base_cfg_file = cfg[BASE_KEY]
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
                # the path to base cfg is relative to the config file itself.
                base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
            base_cfg = cls.load_yaml_with_base(base_cfg_file, allow_unsafe=allow_unsafe)
            del cfg[BASE_KEY]

            # pyre-fixme[6]: Expected `Dict[typing.Any, typing.Any]` for 2nd param
            #  but got `None`.
            merge_a_into_b(cfg, base_cfg)
            return base_cfg
        return cfg

    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = False) -> None:
        """
        Merge configs from a given yaml file.

        Args:
            cfg_filename: the file name of the yaml config.
            allow_unsafe: whether to allow loading the config file with
                `yaml.unsafe_load`.
        """
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

    # Forward the following calls to base, but with a check on the BASE_KEY.
    def merge_from_other_cfg(self, cfg_other: object) -> Callable[[], None]:
        """
        Args:
            cfg_other (CfgNode): configs to merge from.
        """
        assert (
            BASE_KEY not in cfg_other  # pyre-ignore
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_other_cfg(cfg_other)

    def merge_from_list(self, cfg_list: List[object]) -> Callable[[], None]:
        """
        Args:
            cfg_list (list): list of configs to merge from.
        """
        keys = set(cfg_list[0::2])
        assert (
            BASE_KEY not in keys
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_list(cfg_list)

    def __setattr__(self, name: str, val: Any) -> None:  # pyre-ignore
        if name.startswith("COMPUTED_"):
            if name in self:
                old_val = self[name]
                if old_val == val:
                    return
                raise KeyError(
                    "Computed attributed '{}' already exists "
                    "with a different value! old={}, new={}.".format(name, old_val, val)
                )
            self[name] = val
        else:
            super().__setattr__(name, val)
