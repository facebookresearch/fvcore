# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-ignore-all-errors

import copy
import unittest
from typing import Any, Dict

from fvcore.common.param_scheduler import StepParamScheduler


class TestStepScheduler(unittest.TestCase):
    _num_updates = 12

    def _get_valid_config(self) -> Dict[str, Any]:
        return {
            "num_updates": self._num_updates,
            "values": [0.1, 0.01, 0.001, 0.0001],
        }

    def test_invalid_config(self):
        # Invalid num epochs
        config = self._get_valid_config()

        bad_config = copy.deepcopy(config)
        bad_config["num_updates"] = -1
        with self.assertRaises(ValueError):
            StepParamScheduler(**bad_config)

        bad_config["values"] = {"a": "b"}
        with self.assertRaises(ValueError):
            StepParamScheduler(**bad_config)

        bad_config["values"] = []
        with self.assertRaises(ValueError):
            StepParamScheduler(**bad_config)

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = StepParamScheduler(**config)
        schedule = [
            scheduler(epoch_num / self._num_updates)
            for epoch_num in range(self._num_updates)
        ]
        expected_schedule = [
            0.1,
            0.1,
            0.1,
            0.01,
            0.01,
            0.01,
            0.001,
            0.001,
            0.001,
            0.0001,
            0.0001,
            0.0001,
        ]

        self.assertEqual(schedule, expected_schedule)
