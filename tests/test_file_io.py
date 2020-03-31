# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import shutil
import tempfile
import unittest
import uuid
from typing import Optional
from unittest.mock import MagicMock, patch

from fvcore.common.file_io import LazyPath, PathManager, get_cache_dir


class TestNativeIO(unittest.TestCase):
    _tmpdir: Optional[str] = None
    _tmpfile: Optional[str] = None
    _tmpfile_contents = "Hello, World"

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.mkdtemp()
        # pyre-ignore
        with open(os.path.join(cls._tmpdir, "test.txt"), "w") as f:
            cls._tmpfile = f.name
            f.write(cls._tmpfile_contents)
            f.flush()

    @classmethod
    def tearDownClass(cls) -> None:
        # Cleanup temp working dir.
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)  # type: ignore

    def test_open(self) -> None:
        # pyre-ignore
        with PathManager.open(self._tmpfile, "r") as f:
            self.assertEqual(f.read(), self._tmpfile_contents)

    def test_open_args(self) -> None:
        PathManager.set_strict_kwargs_checking(True)
        f = PathManager.open(
            self._tmpfile,  # type: ignore
            mode="r",
            buffering=1,
            encoding="UTF-8",
            errors="ignore",
            newline=None,
            closefd=True,
            opener=None,
        )
        f.close()

    def test_get_local_path(self) -> None:
        self.assertEqual(
            # pyre-ignore
            PathManager.get_local_path(self._tmpfile),
            self._tmpfile,
        )

    def test_exists(self) -> None:
        # pyre-ignore
        self.assertTrue(PathManager.exists(self._tmpfile))
        # pyre-ignore
        fake_path = os.path.join(self._tmpdir, uuid.uuid4().hex)
        self.assertFalse(PathManager.exists(fake_path))

    def test_isfile(self) -> None:
        self.assertTrue(PathManager.isfile(self._tmpfile))  # pyre-ignore
        # This is a directory, not a file, so it should fail
        self.assertFalse(PathManager.isfile(self._tmpdir))  # pyre-ignore
        # This is a non-existing path, so it should fail
        fake_path = os.path.join(self._tmpdir, uuid.uuid4().hex)  # pyre-ignore
        self.assertFalse(PathManager.isfile(fake_path))

    def test_isdir(self) -> None:
        # pyre-ignore
        self.assertTrue(PathManager.isdir(self._tmpdir))
        # This is a file, not a directory, so it should fail
        # pyre-ignore
        self.assertFalse(PathManager.isdir(self._tmpfile))
        # This is a non-existing path, so it should fail
        # pyre-ignore
        fake_path = os.path.join(self._tmpdir, uuid.uuid4().hex)
        self.assertFalse(PathManager.isdir(fake_path))

    def test_ls(self) -> None:
        # Create some files in the tempdir to ls out.
        root_dir = os.path.join(self._tmpdir, "ls")  # pyre-ignore
        os.makedirs(root_dir, exist_ok=True)
        files = sorted(["foo.txt", "bar.txt", "baz.txt"])
        for f in files:
            open(os.path.join(root_dir, f), "a").close()

        children = sorted(PathManager.ls(root_dir))
        self.assertListEqual(children, files)

        # Cleanup the tempdir
        shutil.rmtree(root_dir)

    def test_mkdirs(self) -> None:
        # pyre-ignore
        new_dir_path = os.path.join(self._tmpdir, "new", "tmp", "dir")
        self.assertFalse(PathManager.exists(new_dir_path))
        PathManager.mkdirs(new_dir_path)
        self.assertTrue(PathManager.exists(new_dir_path))

    def test_copy(self) -> None:
        _tmpfile_2 = self._tmpfile + "2"  # pyre-ignore
        _tmpfile_2_contents = "something else"
        with open(_tmpfile_2, "w") as f:
            f.write(_tmpfile_2_contents)
            f.flush()
        # pyre-ignore
        assert PathManager.copy(self._tmpfile, _tmpfile_2, True)
        with PathManager.open(_tmpfile_2, "r") as f:
            self.assertEqual(f.read(), self._tmpfile_contents)

    def test_symlink(self) -> None:
        _symlink = self._tmpfile + "_symlink"  # pyre-ignore
        assert PathManager.symlink(self._tmpfile, _symlink)  # pyre-ignore
        with PathManager.open(_symlink) as f:
            self.assertEqual(f.read(), self._tmpfile_contents)
        assert os.readlink(_symlink) == self._tmpfile
        os.remove(_symlink)

    def test_rm(self) -> None:
        # pyre-ignore
        with open(os.path.join(self._tmpdir, "test_rm.txt"), "w") as f:
            rm_file = f.name
            f.write(self._tmpfile_contents)
            f.flush()
        self.assertTrue(PathManager.exists(rm_file))
        self.assertTrue(PathManager.isfile(rm_file))
        PathManager.rm(rm_file)
        self.assertFalse(PathManager.exists(rm_file))
        self.assertFalse(PathManager.isfile(rm_file))

    def test_bad_args(self) -> None:
        # TODO (T58240718): Replace with dynamic checks
        with self.assertRaises(ValueError):
            PathManager.copy(
                self._tmpfile, self._tmpfile, foo="foo"  # type: ignore
            )
        with self.assertRaises(ValueError):
            PathManager.exists(self._tmpfile, foo="foo")  # type: ignore
        with self.assertRaises(ValueError):
            PathManager.get_local_path(self._tmpfile, foo="foo")  # type: ignore
        with self.assertRaises(ValueError):
            PathManager.isdir(self._tmpfile, foo="foo")  # type: ignore
        with self.assertRaises(ValueError):
            PathManager.isfile(self._tmpfile, foo="foo")  # type: ignore
        with self.assertRaises(ValueError):
            PathManager.ls(self._tmpfile, foo="foo")  # type: ignore
        with self.assertRaises(ValueError):
            PathManager.mkdirs(self._tmpfile, foo="foo")  # type: ignore
        with self.assertRaises(ValueError):
            PathManager.open(self._tmpfile, foo="foo")  # type: ignore
        with self.assertRaises(ValueError):
            PathManager.rm(self._tmpfile, foo="foo")  # type: ignore

        PathManager.set_strict_kwargs_checking(False)

        PathManager.copy(
            self._tmpfile, self._tmpfile, foo="foo"  # type: ignore
        )
        PathManager.exists(self._tmpfile, foo="foo")  # type: ignore
        PathManager.get_local_path(self._tmpfile, foo="foo")  # type: ignore
        PathManager.isdir(self._tmpfile, foo="foo")  # type: ignore
        PathManager.isfile(self._tmpfile, foo="foo")  # type: ignore
        PathManager.ls(self._tmpdir, foo="foo")  # type: ignore
        PathManager.mkdirs(self._tmpdir, foo="foo")  # type: ignore
        f = PathManager.open(self._tmpfile, foo="foo")  # type: ignore
        f.close()
        # pyre-ignore
        with open(os.path.join(self._tmpdir, "test_rm.txt"), "w") as f:
            rm_file = f.name
            f.write(self._tmpfile_contents)
            f.flush()
        PathManager.rm(rm_file, foo="foo")  # type: ignore


class TestHTTPIO(unittest.TestCase):
    _remote_uri = "https://www.facebook.com"
    _filename = "facebook.html"
    _cache_dir: str = os.path.join(get_cache_dir(), __name__)

    def setUp(self) -> None:
        if os.path.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)

    @patch("fvcore.common.file_io.get_cache_dir")
    def test_get_local_path(self, mock_get_cache_dir) -> None:  # pyre-ignore
        mock_get_cache_dir.return_value = self._cache_dir
        local_path = PathManager.get_local_path(self._remote_uri)
        self.assertTrue(os.path.exists(local_path))
        self.assertTrue(os.path.isfile(local_path))

    @patch("fvcore.common.file_io.get_cache_dir")
    def test_open(self, mock_get_cache_dir) -> None:  # pyre-ignore
        mock_get_cache_dir.return_value = self._cache_dir
        with PathManager.open(self._remote_uri, "rb") as f:
            self.assertTrue(os.path.exists(f.name))
            self.assertTrue(os.path.isfile(f.name))
            self.assertTrue(f.read() != "")

    def test_open_writes(self) -> None:
        # HTTPURLHandler does not support writing, only reading.
        with self.assertRaises(AssertionError):
            with PathManager.open(self._remote_uri, "w") as f:
                f.write("foobar")  # pyre-ignore

    def test_bad_args(self) -> None:
        with self.assertRaises(NotImplementedError):
            PathManager.copy(
                self._remote_uri, self._remote_uri, foo="foo"  # type: ignore
            )
        with self.assertRaises(NotImplementedError):
            PathManager.exists(self._remote_uri, foo="foo")  # type: ignore
        with self.assertRaises(ValueError):
            PathManager.get_local_path(
                self._remote_uri, foo="foo"  # type: ignore
            )
        with self.assertRaises(NotImplementedError):
            PathManager.isdir(self._remote_uri, foo="foo")  # type: ignore
        with self.assertRaises(NotImplementedError):
            PathManager.isfile(self._remote_uri, foo="foo")  # type: ignore
        with self.assertRaises(NotImplementedError):
            PathManager.ls(self._remote_uri, foo="foo")  # type: ignore
        with self.assertRaises(NotImplementedError):
            PathManager.mkdirs(self._remote_uri, foo="foo")  # type: ignore
        with self.assertRaises(ValueError):
            PathManager.open(self._remote_uri, foo="foo")  # type: ignore
        with self.assertRaises(NotImplementedError):
            PathManager.rm(self._remote_uri, foo="foo")  # type: ignore

        PathManager.set_strict_kwargs_checking(False)

        PathManager.get_local_path(self._remote_uri, foo="foo")  # type: ignore
        f = PathManager.open(self._remote_uri, foo="foo")  # type: ignore
        f.close()
        PathManager.set_strict_kwargs_checking(True)


class TestLazyPath(unittest.TestCase):
    def test_materialize(self) -> None:
        f = MagicMock(return_value="test")
        x = LazyPath(f)
        f.assert_not_called()

        p = os.fspath(x)
        f.assert_called()
        self.assertEqual(p, "test")

        p = os.fspath(x)
        # should only be called once
        f.assert_called_once()
        self.assertEqual(p, "test")

    def test_join(self) -> None:
        f = MagicMock(return_value="test")
        x = LazyPath(f)
        p = os.path.join(x, "a.txt")
        f.assert_called_once()
        self.assertEqual(p, "test/a.txt")

    def test_getattr(self) -> None:
        x = LazyPath(lambda: "abc")
        with self.assertRaises(AttributeError):
            x.startswith("ab")
        _ = os.fspath(x)
        self.assertTrue(x.startswith("ab"))

    def test_PathManager(self) -> None:
        x = LazyPath(lambda: "./")
        output = PathManager.ls(x)  # pyre-ignore
        output_gt = PathManager.ls("./")
        self.assertEqual(sorted(output), sorted(output_gt))

    def test_getitem(self) -> None:
        x = LazyPath(lambda: "abc")
        with self.assertRaises(TypeError):
            x[0]
        _ = os.fspath(x)
        self.assertEqual(x[0], "a")
