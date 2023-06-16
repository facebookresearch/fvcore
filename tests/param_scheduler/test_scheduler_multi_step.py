# Copyright (c) Facebook, Inc. and its affiliates.

import copy
import unittest

from fvcore.common.param_scheduler import MultiStepParamScheduler


class TestMultiStepParamScheduler(unittest.TestCase):
    _num_updates = 12

    def _get_valid_config(self):
        return {
            "num_updates": self._num_updates,
            "values": [0.1, 0.01, 0.001, 0.0001],
            "milestones": [4, 6, 8],
        }

    def test_invalid_config(self):
        # Invalid num epochs
        config = self._get_valid_config()

        bad_config = copy.deepcopy(config)
        bad_config["num_updates"] = -1
        with self.assertRaises(ValueError):
            MultiStepParamScheduler(**bad_config)

        bad_config["values"] = {"a": "b"}
        with self.assertRaises(ValueError):
            MultiStepParamScheduler(**bad_config)

        bad_config["values"] = []
        with self.assertRaises(ValueError):
            MultiStepParamScheduler(**bad_config)

        # Invalid drop epochs
        bad_config["values"] = config["values"]
        bad_config["milestones"] = {"a": "b"}
        with self.assertRaises(ValueError):
            MultiStepParamScheduler(**bad_config)

        # Too many
        bad_config["milestones"] = [3, 6, 8, 12]
        with self.assertRaises(ValueError):
            MultiStepParamScheduler(**bad_config)

        # Too few
        bad_config["milestones"] = [3, 6]
        with self.assertRaises(ValueError):
            MultiStepParamScheduler(**bad_config)

        # Exceeds num_updates
        bad_config["milestones"] = [3, 6, 12]
        with self.assertRaises(ValueError):
            MultiStepParamScheduler(**bad_config)

        # Out of order
        bad_config["milestones"] = [3, 8, 6]
        with self.assertRaises(ValueError):
            MultiStepParamScheduler(**bad_config)

    def _test_config_scheduler(self, config, expected_schedule):
        scheduler = MultiStepParamScheduler(**config)
        schedule = [
            scheduler(epoch_num / self._num_updates)
            for epoch_num in range(self._num_updates)
        ]
        self.assertEqual(schedule, expected_schedule)

    def test_scheduler(self):
        config = self._get_valid_config()
        expected_schedule = [
            0.1,
            0.1,
            0.1,
            0.1,
            0.01,
            0.01,
            0.001,
            0.001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
        ]
        self._test_config_scheduler(config, expected_schedule)

    def test_default_config(self):
        config = self._get_valid_config()
        default_config = copy.deepcopy(config)
        # Default equispaced drop_epochs behavior
        del default_config["milestones"]
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
        self._test_config_scheduler(default_config, expected_schedule)

    def test_optional_args(self):
        v = [1, 0.1, 0.01]
        s1 = MultiStepParamScheduler(v, num_updates=90, milestones=[30, 60])
        s2 = MultiStepParamScheduler(v, num_updates=90)
        s3 = MultiStepParamScheduler(v, milestones=[30, 60, 90])
        for i in range(10):
            k = i / 10
            self.assertEqual(s1(k), s2(k))
            self.assertEqual(s1(k), s3(k))
