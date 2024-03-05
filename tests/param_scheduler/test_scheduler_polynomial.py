# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-ignore-all-errors

import unittest

from fvcore.common.param_scheduler import PolynomialDecayParamScheduler


class TestPolynomialScheduler(unittest.TestCase):
    _num_epochs = 10

    def test_scheduler(self):
        scheduler = PolynomialDecayParamScheduler(base_value=0.1, power=1)
        schedule = [
            round(scheduler(epoch_num / self._num_epochs), 2)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]

        self.assertEqual(schedule, expected_schedule)
