# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-ignore-all-errors

import copy
import unittest

from fvcore.common.param_scheduler import (
    CompositeParamScheduler,
    ConstantParamScheduler,
    CosineParamScheduler,
    LinearParamScheduler,
    StepParamScheduler,
)


class TestCompositeScheduler(unittest.TestCase):
    _num_updates = 10

    def _get_valid_long_config(self):
        return {
            "schedulers": [
                ConstantParamScheduler(0.1),
                ConstantParamScheduler(0.2),
                ConstantParamScheduler(0.3),
                ConstantParamScheduler(0.4),
            ],
            "lengths": [0.2, 0.4, 0.1, 0.3],
            "interval_scaling": ["rescaled"] * 4,
        }

    def _get_lengths_sum_less_one_config(self):
        return {
            "schedulers": [
                ConstantParamScheduler(0.1),
                ConstantParamScheduler(0.2),
            ],
            "lengths": [0.7, 0.2999],
            "interval_scaling": ["rescaled", "rescaled"],
        }

    def _get_valid_mixed_config(self):
        return {
            "schedulers": [
                StepParamScheduler(values=[0.1, 0.2, 0.3, 0.4, 0.5], num_updates=10),
                CosineParamScheduler(start_value=0.42, end_value=0.0001),
            ],
            "lengths": [0.5, 0.5],
            "interval_scaling": ["rescaled", "rescaled"],
        }

    def _get_valid_linear_config(self):
        return {
            "schedulers": [
                LinearParamScheduler(start_value=0.0, end_value=0.5),
                LinearParamScheduler(start_value=0.5, end_value=1.0),
            ],
            "lengths": [0.5, 0.5],
            "interval_scaling": ["rescaled", "rescaled"],
        }

    def test_invalid_config(self):
        config = self._get_valid_mixed_config()
        bad_config = copy.deepcopy(config)

        # Size of schedulers and lengths doesn't match
        bad_config["schedulers"] = copy.deepcopy(config["schedulers"])
        bad_config["lengths"] = copy.deepcopy(config["lengths"])
        bad_config["schedulers"].append(bad_config["schedulers"][-1])
        with self.assertRaises(ValueError):
            CompositeParamScheduler(**bad_config)

        # Sum of lengths < 1
        bad_config["schedulers"] = copy.deepcopy(config["schedulers"])
        bad_config["lengths"][-1] -= 0.1
        with self.assertRaises(ValueError):
            CompositeParamScheduler(**bad_config)

        # Sum of lengths > 1
        bad_config["lengths"] = copy.deepcopy(config["lengths"])
        bad_config["lengths"][-1] += 0.1
        with self.assertRaises(ValueError):
            CompositeParamScheduler(**bad_config)

        # Bad value for composition_mode
        bad_config["interval_scaling"] = ["rescaled", "rescaleds"]
        with self.assertRaises(ValueError):
            CompositeParamScheduler(**bad_config)

        # Wrong number composition modes
        bad_config["interval_scaling"] = ["rescaled"]
        with self.assertRaises(ValueError):
            CompositeParamScheduler(**bad_config)

    def test_long_scheduler(self):
        config = self._get_valid_long_config()

        scheduler = CompositeParamScheduler(**config)
        schedule = [
            scheduler(epoch_num / self._num_updates)
            for epoch_num in range(self._num_updates)
        ]
        expected_schedule = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.4, 0.4]

        self.assertEqual(schedule, expected_schedule)

    def test_scheduler_lengths_within_epsilon_of_one(self):
        config = self._get_lengths_sum_less_one_config()
        scheduler = CompositeParamScheduler(**config)
        schedule = [
            scheduler(epoch_num / self._num_updates)
            for epoch_num in range(self._num_updates)
        ]
        expected_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
        self.assertEqual(schedule, expected_schedule)

    def test_scheduler_with_mixed_types(self):
        config = self._get_valid_mixed_config()
        scheduler_0 = config["schedulers"][0]
        scheduler_1 = config["schedulers"][1]

        # Check scaled
        config["interval_scaling"] = ["rescaled", "rescaled"]
        scheduler = CompositeParamScheduler(**config)
        scaled_schedule = [
            round(scheduler(epoch_num / self._num_updates), 4)
            for epoch_num in range(self._num_updates)
        ]
        expected_schedule = [
            round(scheduler_0(epoch_num / self._num_updates), 4)
            for epoch_num in range(0, self._num_updates, 2)
        ] + [
            round(scheduler_1(epoch_num / self._num_updates), 4)
            for epoch_num in range(0, self._num_updates, 2)
        ]
        self.assertEqual(scaled_schedule, expected_schedule)

        # Check fixed
        config["interval_scaling"] = ["fixed", "fixed"]
        scheduler = CompositeParamScheduler(**config)
        fixed_schedule = [
            round(scheduler(epoch_num / self._num_updates), 4)
            for epoch_num in range(self._num_updates)
        ]
        expected_schedule = [
            round(scheduler_0(epoch_num / self._num_updates), 4)
            for epoch_num in range(0, int(self._num_updates / 2))
        ] + [
            round(scheduler_1(epoch_num / self._num_updates), 4)
            for epoch_num in range(int(self._num_updates / 2), self._num_updates)
        ]
        self.assertEqual(fixed_schedule, expected_schedule)

        # Check warmup of rescaled then fixed
        config["interval_scaling"] = ["rescaled", "fixed"]
        scheduler = CompositeParamScheduler(**config)
        fixed_schedule = [
            round(scheduler(epoch_num / self._num_updates), 4)
            for epoch_num in range(self._num_updates)
        ]
        expected_schedule = [
            round(scheduler_0(epoch_num / self._num_updates), 4)
            for epoch_num in range(0, int(self._num_updates), 2)
        ] + [
            round(scheduler_1(epoch_num / self._num_updates), 4)
            for epoch_num in range(int(self._num_updates / 2), self._num_updates)
        ]
        self.assertEqual(fixed_schedule, expected_schedule)

    def test_linear_scheduler_no_gaps(self):
        config = self._get_valid_linear_config()

        # Check rescaled
        scheduler = CompositeParamScheduler(**config)
        schedule = [
            scheduler(epoch_num / self._num_updates)
            for epoch_num in range(self._num_updates)
        ]
        expected_schedule = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.assertEqual(expected_schedule, schedule)

        # Check fixed composition gives same result as only 1 scheduler
        config["schedulers"][1] = config["schedulers"][0]
        config["interval_scaling"] = ["fixed", "fixed"]
        scheduler = CompositeParamScheduler(**config)
        linear_scheduler = config["schedulers"][0]
        schedule = [
            scheduler(epoch_num / self._num_updates)
            for epoch_num in range(self._num_updates)
        ]
        expected_schedule = [
            linear_scheduler(epoch_num / self._num_updates)
            for epoch_num in range(self._num_updates)
        ]
        self.assertEqual(expected_schedule, schedule)
