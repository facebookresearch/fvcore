#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import shutil
import tempfile
import unittest
import uuid
from typing import Optional
from unittest.mock import patch

from fvcore.common.file_io import PathManager, get_cache_dir


class TestNativeIO(unittest.TestCase):
    _tmpdir: Optional[str] = None
    _tmpfile: Optional[str] = None
    _tmpfile_contents = "Hello, World"

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()
        with open(os.path.join(cls._tmpdir, "test.txt"), "w") as f:
            cls._tmpfile = f.name
            f.write(cls._tmpfile_contents)
            f.flush()

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)  # type: ignore

    def test_open(self):
        with PathManager.open(self._tmpfile, "r") as f:
            self.assertEqual(f.read(), self._tmpfile_contents)

    def test_get_local_path(self):
        self.assertEqual(
            PathManager.get_local_path(self._tmpfile), self._tmpfile
        )

    def test_exists(self):
        self.assertTrue(PathManager.exists(self._tmpfile))
        fake_path = os.path.join(self._tmpdir, uuid.uuid4().hex)
        self.assertFalse(PathManager.exists(fake_path))

    def test_isfile(self):
        self.assertTrue(PathManager.isfile(self._tmpfile))
        # This is a directory, not a file, so it should fail
        self.assertFalse(PathManager.isfile(self._tmpdir))
        # This is a non-existing path, so it should fail
        fake_path = os.path.join(self._tmpdir, uuid.uuid4().hex)
        self.assertFalse(PathManager.isfile(fake_path))

    def test_isdir(self):
        self.assertTrue(PathManager.isdir(self._tmpdir))
        # This is a file, not a directory, so it should fail
        self.assertFalse(PathManager.isdir(self._tmpfile))
        # This is a non-existing path, so it should fail
        fake_path = os.path.join(self._tmpdir, uuid.uuid4().hex)
        self.assertFalse(PathManager.isdir(fake_path))

    def test_ls(self):
        # Create some files in the tempdir to ls out.
        root_dir = os.path.join(self._tmpdir, "ls")
        os.makedirs(root_dir, exist_ok=True)
        files = sorted(["foo.txt", "bar.txt", "baz.txt"])
        for f in files:
            open(os.path.join(root_dir, f), "a").close()

        children = sorted(PathManager.ls(root_dir))
        self.assertListEqual(children, files)

        # Cleanup the tempdir
        shutil.rmtree(root_dir)

    def test_mkdirs(self):
        new_dir_path = os.path.join(self._tmpdir, "new", "tmp", "dir")
        self.assertFalse(PathManager.exists(new_dir_path))
        PathManager.mkdirs(new_dir_path)
        self.assertTrue(PathManager.exists(new_dir_path))

    def test_copy(self):
        _tmpfile_2 = self._tmpfile + "2"
        _tmpfile_2_contents = "something else"
        with open(_tmpfile_2, "w") as f:
            f.write(_tmpfile_2_contents)
            f.flush()
        assert PathManager.copy(self._tmpfile, _tmpfile_2, True)
        with PathManager.open(_tmpfile_2, "r") as f:
            self.assertEqual(f.read(), self._tmpfile_contents)

    def test_rm(self):
        with open(os.path.join(self._tmpdir, "test_rm.txt"), "w") as f:
            rm_file = f.name
            f.write(self._tmpfile_contents)
            f.flush()
        self.assertTrue(PathManager.exists(rm_file))
        self.assertTrue(PathManager.isfile(rm_file))
        PathManager.rm(rm_file)
        self.assertFalse(PathManager.exists(rm_file))
        self.assertFalse(PathManager.isfile(rm_file))


class TestHTTPIO(unittest.TestCase):
    _remote_uri = "https://www.facebook.com"
    _filename = "facebook.html"
    _cache_dir = os.path.join(get_cache_dir(), __name__)

    def setUp(self) -> None:
        if os.path.exists(self._cache_dir):
            print(f"rmtree {self._cache_dir}")
            shutil.rmtree(self._cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)

    @patch("fvcore.common.file_io.get_cache_dir")
    def test_get_local_path(self, mock_get_cache_dir):
        mock_get_cache_dir.return_value = self._cache_dir
        local_path = PathManager.get_local_path(self._remote_uri)
        self.assertTrue(os.path.exists(local_path))
        self.assertTrue(os.path.isfile(local_path))

    @patch("fvcore.common.file_io.get_cache_dir")
    def test_open(self, mock_get_cache_dir):
        mock_get_cache_dir.return_value = self._cache_dir
        with PathManager.open(self._remote_uri, "r") as f:
            self.assertTrue(os.path.exists(f.name))
            self.assertTrue(os.path.isfile(f.name))
            self.assertTrue(f.read() != "")

    def test_open_writes(self):
        # HTTPURLHandler does not support writing, only reading.
        with self.assertRaises(AssertionError):
            with PathManager.open(self._remote_uri, "w") as f:
                f.write("foobar")
