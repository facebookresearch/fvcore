# Copyright (c) Facebook, Inc. and its affiliates.

# pyre-ignore-all-errors

import unittest

from fvcore.common.param_scheduler import ConstantParamScheduler


class TestConstantScheduler(unittest.TestCase):
    _num_epochs = 12

    def test_scheduler(self):
        scheduler = ConstantParamScheduler(0.1)
        schedule = [
            scheduler(epoch_num / self._num_epochs)
            for epoch_num in range(self._num_epochs)
        ]
        expected_schedule = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        self.assertEqual(schedule, expected_schedule)
        # The input for the scheduler should be in the interval [0;1), open
        with self.assertRaises(RuntimeError):
            scheduler(1)
