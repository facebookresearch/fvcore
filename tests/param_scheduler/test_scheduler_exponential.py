# Copyright (c) Facebook, Inc. and its affiliates.

import unittest

from fvcore.common.param_scheduler import ExponentialParamScheduler


class TestExponentialScheduler(unittest.TestCase):
    _num_epochs = 10

    def _get_valid_config(self):
        return {"start_value": 2.0, "decay": 0.1}

    def _get_valid_intermediate_values(self):
        return [1.5887, 1.2619, 1.0024, 0.7962, 0.6325, 0.5024, 0.3991, 0.3170, 0.2518]

    def test_scheduler(self):
        config = self._get_valid_config()

        scheduler = ExponentialParamScheduler(**config)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 4)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [
            config["start_value"]
        ] + self._get_valid_intermediate_values()

        self.assertEqual(schedule, expected_schedule)
