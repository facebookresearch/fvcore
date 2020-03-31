# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import unittest

from fvcore.common.download import download


class TestDownload(unittest.TestCase):
    _filename = "facebook.html"

    def test_download(self) -> None:
        download(
            "https://www.facebook.com",
            ".",
            filename=self._filename,
            progress=False,
        )
        self.assertTrue(os.path.isfile(self._filename))

    def tearDown(self) -> None:
        if os.path.isfile(self._filename):
            os.unlink(self._filename)
