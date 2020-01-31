#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import copy
import os
import random
import string
import unittest
from collections import OrderedDict
from tempfile import TemporaryDirectory
import torch
from torch import nn

from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer


class TestCheckpointer(unittest.TestCase):
    def _create_model(self):
        """
        Create a simple model.
        """
        return nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

    def _create_complex_model(self):
        """
        Create a complex model.
        """
        m = nn.Module()
        m.block1 = nn.Module()
        m.block1.layer1 = nn.Linear(2, 3)
        m.layer2 = nn.Linear(3, 2)
        m.res = nn.Module()
        m.res.layer2 = nn.Linear(3, 2)

        state_dict = OrderedDict()
        state_dict["layer1.weight"] = torch.rand(3, 2)
        state_dict["layer1.bias"] = torch.rand(3)
        state_dict["layer2.weight"] = torch.rand(2, 3)
        state_dict["layer2.bias"] = torch.rand(2)
        state_dict["res.layer2.weight"] = torch.rand(2, 3)
        state_dict["res.layer2.bias"] = torch.rand(2)

        return m, state_dict

    def test_from_last_checkpoint_model(self):
        """
        test that loading works even if they differ by a prefix.
        """
        for trained_model, fresh_model in [
            (self._create_model(), self._create_model()),
            (nn.DataParallel(self._create_model()), self._create_model()),
            (self._create_model(), nn.DataParallel(self._create_model())),
            (
                nn.DataParallel(self._create_model()),
                nn.DataParallel(self._create_model()),
            ),
        ]:

            with TemporaryDirectory() as f:
                checkpointer = Checkpointer(trained_model, save_dir=f)
                checkpointer.save("checkpoint_file")

                # in the same folder
                fresh_checkpointer = Checkpointer(fresh_model, save_dir=f)
                self.assertTrue(fresh_checkpointer.has_checkpoint())
                self.assertEqual(
                    fresh_checkpointer.get_checkpoint_file(),
                    os.path.join(f, "checkpoint_file.pth"),
                )
                fresh_checkpointer.load(
                    fresh_checkpointer.get_checkpoint_file()
                )

            for trained_p, loaded_p in zip(
                trained_model.parameters(), fresh_model.parameters()
            ):
                # different tensor references
                self.assertFalse(id(trained_p) == id(loaded_p))
                # same content
                self.assertTrue(trained_p.cpu().equal(loaded_p.cpu()))

    def test_from_name_file_model(self):
        """
        test that loading works even if they differ by a prefix.
        """
        for trained_model, fresh_model in [
            (self._create_model(), self._create_model()),
            (nn.DataParallel(self._create_model()), self._create_model()),
            (self._create_model(), nn.DataParallel(self._create_model())),
            (
                nn.DataParallel(self._create_model()),
                nn.DataParallel(self._create_model()),
            ),
        ]:
            with TemporaryDirectory() as f:
                checkpointer = Checkpointer(
                    trained_model, save_dir=f, save_to_disk=True
                )
                checkpointer.save("checkpoint_file")

                # on different folders.
                with TemporaryDirectory() as g:
                    fresh_checkpointer = Checkpointer(fresh_model, save_dir=g)
                    self.assertFalse(fresh_checkpointer.has_checkpoint())
                    self.assertEqual(
                        fresh_checkpointer.get_checkpoint_file(), ""
                    )
                    fresh_checkpointer.load(
                        os.path.join(f, "checkpoint_file.pth")
                    )

            for trained_p, loaded_p in zip(
                trained_model.parameters(), fresh_model.parameters()
            ):
                # different tensor references.
                self.assertFalse(id(trained_p) == id(loaded_p))
                # same content.
                self.assertTrue(trained_p.cpu().equal(loaded_p.cpu()))

    def test_checkpointables(self):
        """
        Test saving and loading checkpointables.
        """

        class CheckpointableObj:
            """
            A dummy checkpointableObj class with state_dict and load_state_dict
            methods.
            """

            def __init__(self):
                self.state = {
                    self.random_handle(): self.random_handle()
                    for i in range(10)
                }

            def random_handle(self, str_len=100):
                """
                Generate a random string of fixed length.
                Args:
                    str_len (str): length of the output string.
                Returns:
                    (str): random generated handle.
                """
                letters = string.ascii_uppercase
                return "".join(random.choice(letters) for i in range(str_len))

            def state_dict(self):
                """
                Return the state.
                Returns:
                    (dict): return the state.
                """
                return self.state

            def load_state_dict(self, state):
                """
                Load the state from a given state.
                Args:
                    state (dict): a key value dictionary.
                """
                self.state = copy.deepcopy(state)

        trained_model, fresh_model = self._create_model(), self._create_model()
        with TemporaryDirectory() as f:
            checkpointables = CheckpointableObj()
            checkpointer = Checkpointer(
                trained_model,
                save_dir=f,
                save_to_disk=True,
                checkpointables=checkpointables,
            )
            checkpointer.save("checkpoint_file")
            # in the same folder
            fresh_checkpointer = Checkpointer(fresh_model, save_dir=f)
            self.assertTrue(fresh_checkpointer.has_checkpoint())
            self.assertEqual(
                fresh_checkpointer.get_checkpoint_file(),
                os.path.join(f, "checkpoint_file.pth"),
            )
            checkpoint = fresh_checkpointer.load(
                fresh_checkpointer.get_checkpoint_file()
            )
            state_dict = checkpointables.state_dict()
            for key, _ in state_dict.items():
                self.assertTrue(
                    checkpoint["checkpointables"].get(key) is not None
                )
                self.assertTrue(
                    checkpoint["checkpointables"][key] == state_dict[key]
                )


class TestPeriodicCheckpointer(unittest.TestCase):
    def _create_model(self):
        """
        Create a simple model.
        """
        return nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))

    def test_periodic_checkpointer(self):
        """
        test that loading works even if they differ by a prefix.
        """
        _period = 10
        _max_iter = 100
        for trained_model in [
            self._create_model(),
            nn.DataParallel(self._create_model()),
        ]:
            with TemporaryDirectory() as f:
                checkpointer = Checkpointer(
                    trained_model, save_dir=f, save_to_disk=True
                )
                periodic_checkpointer = PeriodicCheckpointer(
                    checkpointer, _period, 99
                )
                for iteration in range(_max_iter):
                    periodic_checkpointer.step(iteration)
                    path = os.path.join(f, "model_{:07d}.pth".format(iteration))
                    if (iteration + 1) % _period == 0:
                        self.assertTrue(os.path.exists(path))
                    else:
                        self.assertFalse(os.path.exists(path))

    def test_periodic_checkpointer_max_to_keep(self):
        """
        Test parameter: max_to_keep
        """
        _period = 10
        _max_iter = 100
        _max_to_keep = 3
        for trained_model in [
            self._create_model(),
            nn.DataParallel(self._create_model()),
        ]:
            with TemporaryDirectory() as f:
                checkpointer = Checkpointer(
                    trained_model, save_dir=f, save_to_disk=True
                )
                periodic_checkpointer = PeriodicCheckpointer(
                    checkpointer, _period, 99, max_to_keep=_max_to_keep
                )

                checkpoint_paths = []

                for iteration in range(_max_iter):
                    periodic_checkpointer.step(iteration)
                    if (iteration + 1) % _period == 0:
                        path = os.path.join(
                            f, "model_{:07d}.pth".format(iteration)
                        )
                        checkpoint_paths.append(path)

                for path in checkpoint_paths[:-_max_to_keep]:
                    self.assertFalse(os.path.exists(path))

                for path in checkpoint_paths[-_max_to_keep:]:
                    self.assertTrue(os.path.exists(path))
