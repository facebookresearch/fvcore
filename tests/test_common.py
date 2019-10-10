#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy as np
import time
import unittest

from fvcore.common.config import CfgNode
from fvcore.common.history_buffer import HistoryBuffer
from fvcore.common.timer import Timer


class TestHistoryBuffer(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)

    @staticmethod
    def create_buffer_with_init(num_values: int, buffer_len: int = 1000000):
        """
        Return a HistoryBuffer of the given length filled with random numbers.

        Args:
            buffer_len: length of the created history buffer.
            num_values: number of random numbers added to the history buffer.
        """

        max_value = 1000
        values = np.random.randint(max_value, size=num_values)

        def create_buffer():
            buf = HistoryBuffer(buffer_len)
            for v in values:
                buf.update(v)
            return buf, values

        return create_buffer

    def test_buffer(self) -> None:
        """
        Test creation of HistoryBuffer and the methods provided in the class.
        """

        num_iters = 100
        for _ in range(num_iters):
            gt_len = 1000
            buffer_len = np.random.randint(1, gt_len)
            create_buffer = TestHistoryBuffer.create_buffer_with_init(
                gt_len, buffer_len
            )
            buf, gt = create_buffer()

            values, iterations = zip(*buf.values())
            self.assertEqual(len(values), buffer_len)
            self.assertEqual(len(iterations), buffer_len)
            self.assertTrue((values == gt[-buffer_len:]).all())
            iterations_gt = np.arange(gt_len - buffer_len, gt_len)
            self.assertTrue(
                (iterations == iterations_gt).all(),
                ", ".join(str(x) for x in iterations),
            )
            self.assertAlmostEqual(buf.global_avg(), gt.mean())
            w = 100
            effective_w = min(w, buffer_len)
            self.assertAlmostEqual(
                buf.median(w),
                np.median(gt[-effective_w:]),
                None,
                " ".join(str(x) for x in gt[-effective_w:]),
            )
            self.assertAlmostEqual(
                buf.avg(w),
                np.mean(gt[-effective_w:]),
                None,
                " ".join(str(x) for x in gt[-effective_w:]),
            )


class TestTimer(unittest.TestCase):
    def test_timer(self):
        timer = Timer()
        time.sleep(0.5)
        self.assertTrue(0.99 > timer.seconds() >= 0.5)

        timer.pause()
        time.sleep(0.5)

        self.assertTrue(0.99 > timer.seconds() >= 0.5)

        timer.resume()
        time.sleep(0.5)
        self.assertTrue(1.49 > timer.seconds() >= 1.0)

        timer.reset()
        self.assertTrue(0.49 > timer.seconds() >= 0)


class TestCfgNode(unittest.TestCase):
    @staticmethod
    def gen_default_cfg():
        cfg = CfgNode()
        cfg.KEY1 = "default"
        cfg.KEY2 = "default"
        cfg.EXPRESSION = [3.0]

        return cfg

    def test_merge_from_file(self):
        """
        Test merge_from_file function provided in the class.
        """
        import pkg_resources

        base_yaml = pkg_resources.resource_filename(
            __name__, "configs/base.yaml"
        )
        config_yaml = pkg_resources.resource_filename(
            __name__, "configs/config.yaml"
        )

        cfg = TestCfgNode.gen_default_cfg()
        cfg.merge_from_file(base_yaml)
        self.assertEqual(cfg.KEY1, "base")
        self.assertEqual(cfg.KEY2, "base")

        cfg = TestCfgNode.gen_default_cfg()

        with self.assertRaises(Exception):
            # config.yaml contains unsafe yaml tags,
            # test if an exception is thrown
            cfg.merge_from_file(config_yaml)

        cfg.merge_from_file(config_yaml, allow_unsafe=True)
        self.assertEqual(cfg.KEY1, "base")
        self.assertEqual(cfg.KEY2, "config")
        self.assertEqual(cfg.EXPRESSION, [1, 4, 9])

    def test_merge_from_list(self):
        """
        Test merge_from_list function provided in the class.
        """
        cfg = TestCfgNode.gen_default_cfg()
        cfg.merge_from_list(["KEY1", "list1", "KEY2", "list2"])
        self.assertEqual(cfg.KEY1, "list1")
        self.assertEqual(cfg.KEY2, "list2")

    def test_setattr(self):
        """
        Test __setattr__ function provided in the class.
        """
        cfg = TestCfgNode.gen_default_cfg()
        cfg.KEY1 = "new1"
        cfg.KEY3 = "new3"
        self.assertEqual(cfg.KEY1, "new1")
        self.assertEqual(cfg.KEY3, "new3")

        # Test computed attributes, which can be inserted regardless of whether
        # the CfgNode is frozen or not.
        cfg = TestCfgNode.gen_default_cfg()
        cfg.COMPUTED_1 = "computed1"
        self.assertEqual(cfg.COMPUTED_1, "computed1")
        cfg.freeze()
        cfg.COMPUTED_2 = "computed2"
        self.assertEqual(cfg.COMPUTED_2, "computed2")

        # Test computed attributes, which should be 'insert only' (could not be
        # updated).
        cfg = TestCfgNode.gen_default_cfg()
        cfg.COMPUTED_1 = "computed1"
        with self.assertRaises(KeyError) as err:
            cfg.COMPUTED_1 = "update_computed1"
        self.assertTrue(
            "Computed attributed 'COMPUTED_1' already exists"
            in str(err.exception)
        )

        # Resetting the same value should be safe:
        cfg.COMPUTED_1 = "computed1"
