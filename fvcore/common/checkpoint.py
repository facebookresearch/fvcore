# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# pyre-ignore-all-errors[2,3,58]

import logging
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from iopath.common.file_io import HTTPURLHandler, PathManager
from termcolor import colored
from torch.nn.parallel import DataParallel, DistributedDataParallel


__all__ = ["Checkpointer", "PeriodicCheckpointer"]


TORCH_VERSION: Tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])


class _IncompatibleKeys(
    NamedTuple(
        "IncompatibleKeys",
        [
            ("missing_keys", List[str]),
            ("unexpected_keys", List[str]),
            ("incorrect_shapes", List[Tuple[str, Tuple[int], Tuple[int]]]),
        ],
    )
):
    pass


class Checkpointer:
    """
    A checkpointer that can save/load model as well as extra checkpointable
    objects.
    """

    def __init__(
        self,
        model: nn.Module,
        save_dir: str = "",
        *,
        save_to_disk: bool = True,
        **checkpointables: Any,
    ) -> None:
        """
        Args:
            model (nn.Module): model.
            save_dir (str): a directory to save and find checkpoints.
            save_to_disk (bool): if True, save checkpoint to disk, otherwise
                disable saving for this checkpointer.
            checkpointables (object): any checkpointable objects, i.e., objects
                that have the ``state_dict()`` and ``load_state_dict()`` method. For
                example, it can be used like
                `Checkpointer(model, "dir", optimizer=optimizer)`.
        """
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables: Dict[str, Any] = {}
        for k, v in checkpointables.items():
            self.add_checkpointable(k, v)
        self.logger: logging.Logger = logging.getLogger(__name__)
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        # Default PathManager, support HTTP URLs (for backward compatibility in open source).
        # A user may want to use a different project-specific PathManager
        self.path_manager: PathManager = PathManager()
        self.path_manager.register_handler(HTTPURLHandler())

    def add_checkpointable(self, key: str, checkpointable: Any) -> None:
        """
        Add checkpointable object for this checkpointer to track.

        Args:
            key (str): the key used to save the object
            checkpointable: any object with ``state_dict()`` and
                ``load_state_dict()`` method
        """
        if key in self.checkpointables:
            raise KeyError(f"Key {key} already used in the Checkpointer")
        if not hasattr(checkpointable, "state_dict"):
            raise TypeError(
                "add_checkpointable needs an object with 'state_dict()' method."
            )
        self.checkpointables[key] = checkpointable

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(
        self, path: str, checkpointables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Load from the given checkpoint.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("[Checkpointer] Loading from {} ...".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if (
            incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(self.save_dir, last_saved)  # pyre-fixme[6]

    def get_all_checkpoint_files(self) -> List[str]:
        """
        Returns:
            list: All available checkpoint files (.pth files) in target
                directory.
        """
        all_model_checkpoints = [
            os.path.join(self.save_dir, file)
            for file in self.path_manager.ls(self.save_dir)
            if self.path_manager.isfile(os.path.join(self.save_dir, file))
            and file.endswith(".pth")
        ]
        return all_model_checkpoints

    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists
                and load the model together with all the checkpointables. Otherwise
                only load the model without loading any checkpointables.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            return self.load(path, checkpointables=[])

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore

    def _load_file(self, f: str) -> Dict[str, Any]:
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.

        Args:
            f (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to torch.Tensor or numpy arrays.
        """
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint: Any) -> _IncompatibleKeys:
        """
        Load weights from a checkpoint.

        Args:
            checkpoint (Any): checkpoint contains the weights.

        Returns:
            ``NamedTuple`` with ``missing_keys``, ``unexpected_keys``,
                and ``incorrect_shapes`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys
                * **incorrect_shapes** is a list of (key, shape in checkpoint, shape in model)

            This is just like the return value of
            :func:`torch.nn.Module.load_state_dict`, but with extra support
            for ``incorrect_shapes``.
        """
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # workaround https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                model_param = model_state_dict[k]
                # Allow mismatch for uninitialized parameters
                if TORCH_VERSION >= (1, 8) and isinstance(
                    model_param, nn.parameter.UninitializedParameter
                ):
                    continue
                shape_model = tuple(model_param.shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:

                    has_observer_base_classes = (
                        TORCH_VERSION >= (1, 8)
                        and hasattr(torch.quantization, "ObserverBase")
                        and hasattr(torch.quantization, "FakeQuantizeBase")
                    )
                    if has_observer_base_classes:
                        # Handle the special case of quantization per channel observers,
                        # where buffer shape mismatches are expected.
                        def _get_module_for_key(
                            model: torch.nn.Module, key: str
                        ) -> torch.nn.Module:
                            # foo.bar.param_or_buffer_name -> [foo, bar]
                            key_parts = key.split(".")[:-1]
                            cur_module = model
                            for key_part in key_parts:
                                cur_module = getattr(cur_module, key_part)
                            return cur_module

                        cls_to_skip = (
                            torch.quantization.ObserverBase,
                            torch.quantization.FakeQuantizeBase,
                        )
                        target_module = _get_module_for_key(self.model, k)
                        if isinstance(target_module, cls_to_skip):
                            # Do not remove modules with expected shape mismatches
                            # them from the state_dict loading. They have special logic
                            # in _load_from_state_dict to handle the mismatches.
                            continue

                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        # pyre-ignore
        incompatible = self.model.load_state_dict(checkpoint_state_dict, strict=False)
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )

    def _log_incompatible_keys(self, incompatible: _IncompatibleKeys) -> None:
        """
        Log information about the incompatible keys returned by ``_load_model``.
        """
        for k, shape_checkpoint, shape_model in incompatible.incorrect_shapes:
            self.logger.warning(
                "Skip loading parameter '{}' to the model due to incompatible "
                "shapes: {} in the checkpoint but {} in the "
                "model! You might want to double check if this is expected.".format(
                    k, shape_checkpoint, shape_model
                )
            )
        if incompatible.missing_keys:
            missing_keys = _filter_reused_missing_keys(
                self.model, incompatible.missing_keys
            )
            if missing_keys:
                self.logger.warning(get_missing_parameters_message(missing_keys))
        if incompatible.unexpected_keys:
            self.logger.warning(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    def _convert_ndarray_to_tensor(self, state_dict: Dict[str, Any]) -> None:
        """
        In-place convert all numpy arrays in the state_dict to torch tensor.
        Args:
            state_dict (dict): a state-dict to be loaded to the model.
                Will be modified.
        """
        # model could be an OrderedDict with _metadata attribute
        # (as returned by Pytorch's state_dict()). We should preserve these
        # properties.
        for k in list(state_dict.keys()):
            v = state_dict[k]
            if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
                raise ValueError(
                    "Unsupported type found in checkpoint! {}: {}".format(k, type(v))
                )
            if not isinstance(v, torch.Tensor):
                state_dict[k] = torch.from_numpy(v)


class PeriodicCheckpointer:
    """
    Save checkpoints periodically. When `.step(iteration)` is called, it will
    execute `checkpointer.save` on the given checkpointer, if iteration is a
    multiple of period or if `max_iter` is reached.

    Attributes:
        checkpointer (Checkpointer): the underlying checkpointer object
    """

    def __init__(
        self,
        checkpointer: Checkpointer,
        period: int,
        max_iter: Optional[int] = None,
        max_to_keep: Optional[int] = None,
        file_prefix: str = "model",
    ) -> None:
        """
        Args:
            checkpointer: the checkpointer object used to save checkpoints.
            period (int): the period to save checkpoint.
            max_iter (int): maximum number of iterations. When it is reached,
                a checkpoint named "{file_prefix}_final" will be saved.
            max_to_keep (int): maximum number of most current checkpoints to keep,
                previous checkpoints will be deleted
            file_prefix (str): the prefix of checkpoint's filename
        """
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_iter = max_iter
        if max_to_keep is not None:
            assert max_to_keep > 0
        self.max_to_keep = max_to_keep
        self.recent_checkpoints: List[str] = []
        self.path_manager: PathManager = checkpointer.path_manager
        self.file_prefix = file_prefix

    def step(self, iteration: int, **kwargs: Any) -> None:
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if (iteration + 1) % self.period == 0:
            self.checkpointer.save(
                "{}_{:07d}".format(self.file_prefix, iteration), **additional_state
            )

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(
                        file_to_delete
                    ) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        self.path_manager.rm(file_to_delete)

        if self.max_iter is not None:
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)

    def save(self, name: str, **kwargs: Any) -> None:
        """
        Same argument as :meth:`Checkpointer.save`.
        Use this method to manually save checkpoints outside the schedule.

        Args:
            name (str): file name.
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        self.checkpointer.save(name, **kwargs)


def _filter_reused_missing_keys(model: nn.Module, keys: List[str]) -> List[str]:
    """
    Filter "missing keys" to not include keys that have been loaded with another name.
    """
    keyset = set(keys)
    param_to_names = defaultdict(set)  # param -> names that points to it
    for module_prefix, module in _named_modules_with_dup(model):
        for name, param in list(module.named_parameters(recurse=False)) + list(
            module.named_buffers(recurse=False)
        ):
            full_name = (module_prefix + "." if module_prefix else "") + name
            param_to_names[param].add(full_name)
    for names in param_to_names.values():
        # if one name appears missing but its alias exists, then this
        # name is not considered missing
        if any(n in keyset for n in names) and not all(n in keyset for n in names):
            [keyset.remove(n) for n in names if n in keyset]
    return list(keyset)


def get_missing_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg_per_group = sorted(k + _group_to_str(v) for k, v in groups.items())
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join([colored(x, "blue") for x in msg_per_group])
    return msg


def get_unexpected_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
    )
    return msg


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> None:
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata  # pyre-ignore
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1 :]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) -> str:
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(sorted(group)) + "}"


def _named_modules_with_dup(
    model: nn.Module, prefix: str = ""
) -> Iterable[Tuple[str, nn.Module]]:
    """
    The same as `model.named_modules()`, except that it includes
    duplicated modules that have more than one name.
    """
    yield prefix, model
    for name, module in model._modules.items():  # pyre-ignore
        if module is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        yield from _named_modules_with_dup(module, submodule_prefix)
