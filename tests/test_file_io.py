# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import inspect
import io
import os
import shutil
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from typing import IO, Generator, Optional, Union
from unittest.mock import MagicMock, patch

from fvcore.common import file_io
from fvcore.common.file_io import (
    GoogleCloudHandler,
    HTTPURLHandler,
    LazyPath,
    PathManager,
    PathManagerBase,
    close_and_upload,
    get_cache_dir,
)
from google.cloud import storage


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

    @contextmanager
    def _patch_download(self) -> Generator[None, None, None]:
        def fake_download(url: str, dir: str, *, filename: str) -> str:
            dest = os.path.join(dir, filename)
            with open(dest, "w") as f:
                f.write("test")
            return dest

        with patch.object(
            file_io, "get_cache_dir", return_value=self._cache_dir
        ), patch.object(file_io, "download", side_effect=fake_download):
            yield

    def setUp(self) -> None:
        if os.path.exists(self._cache_dir):
            shutil.rmtree(self._cache_dir)
        os.makedirs(self._cache_dir, exist_ok=True)

    def test_get_local_path(self) -> None:
        with self._patch_download():
            local_path = PathManager.get_local_path(self._remote_uri)
            self.assertTrue(os.path.exists(local_path))
            self.assertTrue(os.path.isfile(local_path))

    def test_open(self) -> None:
        with self._patch_download():
            with PathManager.open(self._remote_uri, "rb") as f:
                self.assertTrue(os.path.exists(f.name))
                self.assertTrue(os.path.isfile(f.name))
                self.assertTrue(f.read() != "")

    def test_open_writes(self) -> None:
        # HTTPURLHandler does not support writing, only reading.
        with self.assertRaises(AssertionError):
            with PathManager.open(self._remote_uri, "w") as f:
                f.write("foobar")  # pyre-ignore

    def test_open_new_path_manager(self) -> None:
        with self._patch_download():
            path_manager = PathManagerBase()
            with self.assertRaises(OSError):  # no handler registered
                f = path_manager.open(self._remote_uri, "rb")

            path_manager.register_handler(HTTPURLHandler())
            with path_manager.open(self._remote_uri, "rb") as f:
                self.assertTrue(os.path.isfile(f.name))
                self.assertTrue(f.read() != "")

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


class TestOneDrive(unittest.TestCase):
    _url = "https://1drv.ms/u/s!Aus8VCZ_C_33gQbJsUPTIj3rQu99"

    def test_one_drive_download(self) -> None:
        from fvcore.common.file_io import OneDrivePathHandler

        _direct_url = OneDrivePathHandler().create_one_drive_direct_download(self._url)
        _gt_url = (
            "https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBd"
            + "XM4VkNaX0NfMzNnUWJKc1VQVElqM3JRdTk5/root/content"
        )
        self.assertEquals(_direct_url, _gt_url)

class TestCloudUtils(unittest.TestCase):
    gc_auth = False
    skip_gc_auth_required_tests_message = (
        "Provide a GC project and bucket you are authorised against, then set the fc_auth flag to True")

    @classmethod
    def setUpClass(self):
        self.gc_project_name = 'project-name'
        self.gc_bucket_name = 'project-name-data'
        self.gc_default_path = '/'.join(['gs:/', self.gc_bucket_name, 'test'])
        self.gc_pathhandler = GoogleCloudHandler()
    @classmethod
    def tearDownClass(self, _gc_auth=gc_auth):
        shutil.rmtree('tmp/')
        if not _gc_auth: return
        remote_file_path = '/'.join([self.gc_default_path, 'path/test.txt'])
        self.gc_pathhandler._delete_remote_resource(remote_file_path)
        remote_file_path = '/'.join([self.gc_default_path, 'path/uploaded.txt'])
        self.gc_pathhandler._delete_remote_resource(remote_file_path)

    def test_supported_prefixes(self):
        supported_prefixes = self.gc_pathhandler._get_supported_prefixes()
        self.assertEqual(supported_prefixes, ["gs://"])
    
    def test_remove_file_system_from_remote_path(self):
        path = self.gc_pathhandler._remove_file_system('/'.join([self.gc_default_path, 'path/file.txt']))
        self.assertEqual(path, '/'.join([self.gc_bucket_name, 'test/path/file.txt']))
    def test_remove_bucket_name_from_remote_path(self):
        path = self.gc_pathhandler._remove_bucket_name('/'.join([self.gc_default_path, 'path/file.txt']))
        self.assertEqual(path, "gs://test/path/file.txt")
    def test_extract_namespace_from_remote_path(self):
        namespace = self.gc_pathhandler._extract_gc_namespace('/'.join([self.gc_default_path, 'path/file.txt']))
        self.assertEqual(namespace, self.gc_project_name)
    def test_extract_bucket_from_remote_path(self):
        bucket_name = self.gc_pathhandler._extract_gc_bucket_name('/'.join([self.gc_default_path, 'path/file.txt']))
        self.assertEqual(bucket_name, self.gc_bucket_name)
    def test_extract_blob_path(self):
        blob_path = self.gc_pathhandler._extract_blob_path('/'.join([self.gc_default_path, 'path/file.txt']))
        self.assertEqual(blob_path, "test/path/file.txt")
    def test_get_local_cache_path(self):
        tmp_path = self.gc_pathhandler._get_local_cache_path('/'.join([self.gc_default_path, 'path/file.txt']))
        self.assertEqual(tmp_path, "./tmp/test/path/file.txt")
    def test_get_local_cache_directory(self):
        tmp_path = self.gc_pathhandler._get_local_cache_directory('/'.join([self.gc_default_path, 'path/file.txt']))
        self.assertEqual(tmp_path, "./tmp/test/path/")
    
    def _add_gc_methods_to_file(self, file: Union[IO[str], IO[bytes]]):
        gc_blob = storage.Blob('test', storage.Bucket('test'))
        self.gc_pathhandler._decorate_file_with_gc_methods(file, gc_blob)
        self.assertTrue(isinstance(file._gc_blob, storage.Blob))
        self.assertEqual(inspect.getsource(file.close), inspect.getsource(close_and_upload))
        file._close()
        self.assertRaises(ValueError, file.readline)
    def test_maybe_make_directory_doesnt_exist(self):
        self.assertTrue(self.gc_pathhandler._maybe_make_directory("./tmp/test/path/test.txt"))
    def test_maybe_make_directory_exists(self):
        self.assertFalse(self.gc_pathhandler._maybe_make_directory("./tmp/test/path/test.txt"))
    def test_add_gc_methods_to_text_file(self):
        file = open('/tmp/test.txt', 'w')
        self._add_gc_methods_to_file(file)
    def test_add_gc_methods_to_binary_file(self):
        file = open('/tmp/test.txt', 'wb')
        self._add_gc_methods_to_file(file)

    def test_is_file_when_path_is_a_file(self):
        remote_path = '/'.join([self.gc_default_path, 'path/test.txt'])
        is_file = self.gc_pathhandler._isfile
        self.assertTrue(is_file)
    def test_is_file_when_path_is_directory(self):
        remote_path = '/'.join([self.gc_default_path, 'path/'])
        is_file = self.gc_pathhandler._isfile(remote_path)
        self.assertFalse(is_file) 
    def test_is_dir_when_path_is_a_driectory(self):
        remote_path = '/'.join([self.gc_default_path, 'path/'])
        is_directory = self.gc_pathhandler._isdir(remote_path)
        self.assertTrue(is_directory)
    def test_id_dir_when_path_is_a_file(self):
        remote_path = '/'.join([self.gc_default_path, 'path/test.txt'])
        is_directory = self.gc_pathhandler._isdir(remote_path)
        self.assertFalse(is_directory)
    
    # Require GCS Authentication ====>
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_add_client_to_handler(self):
        self.gc_pathhandler._create_gc_client('/'.join([self.gc_default_path, 'path/file.txt']))
        self.assertTrue(isinstance(self.gc_pathhandler._gc_client, storage.Client))
        self.assertEqual(self.gc_pathhandler._gc_client.project, self.gc_project_name)
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_get_requested_gc_bucket(self):
        gc_bucket = self.gc_pathhandler._get_gc_bucket('/'.join([self.gc_default_path, 'path/file.txt']))
        self.assertTrue(isinstance(gc_bucket, storage.Bucket))
        self.assertEqual(gc_bucket.name, self.gc_bucket_name)
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_get_blob(self):
        gc_blob = self.gc_pathhandler._get_blob('/'.join([self.gc_default_path, 'path/file.txt']))
        self.assertTrue(isinstance(gc_blob, storage.Blob))
        self.assertEqual(gc_blob.name, "test/path/file.txt")
    
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_exist_when_blob_exists(self):
        self.assertTrue(self.gc_pathhandler._exists('/'.join([self.gc_default_path, ''])))
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_exist_when_blob_doesnt_exist(self):
        self.assertFalse(self.gc_pathhandler._exists('/'.join([self.gc_default_path, 'doesnt/exist.txt'])))
    
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def _gc_local_file_write_and_upload(self, file: Union[IO[str], IO[bytes]], message: str):
        gc_blob = self.gc_pathhandler._get_blob('/'.join([self.gc_default_path, 'path/test.txt']))
        self.gc_pathhandler._decorate_file_with_gc_methods(file, gc_blob)
        file.write(message)
        file.close()
        self.assertTrue(gc_blob.exists())
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_gc_local_file_binary_write_and_upload(self):
        file = open('/tmp/text_binary.txt', 'wb')
        self._gc_local_file_write_and_upload(file, b'{\x03\xff\x00d')
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_gc_local_file_text_write_and_upload(self):
        file = open('/tmp/test.txt', 'w')
        self._gc_local_file_write_and_upload(file, "This is a google cloud file test\n")
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_open_read_text_file(self):
        file = self.gc_pathhandler._open('/'.join([self.gc_default_path, 'path/test2.txt']))
        self.assertTrue(isinstance(file, io.TextIOWrapper))
        self.assertEqual(file.read(), "Retrieved from GC")
        file.close()
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def write_message_with_open(self, path:str, message:str, mode:str):
        file = self.gc_pathhandler._open(path, mode)
        file.write(message)
        file.close()
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def read_remote_file(self, path:str, mode:str) -> str:
        with self.gc_pathhandler._open(path, mode) as file:
            read = file.read()
        return read
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_open_write_new_text_file(self):
        remote_path = '/'.join([self.gc_default_path, 'path/test_open_write.txt'])
        message = 'File created locally and uploaded with _open'
        self.write_message_with_open(remote_path, message, 'w')
        read = self.read_remote_file(remote_path, 'r')
        self.assertEqual(read, 'File created locally and uploaded with _open')
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_open_write_existing_text_file(self):
        remote_path = '/'.join([self.gc_default_path, 'path/test_open_write.txt'])
        message = 'Written to existing upload'
        self.write_message_with_open(remote_path, message, 'w')
        read = self.read_remote_file(remote_path, 'r')
        self.assertEqual(read, 'Written to existing upload')

    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_copy_from_local_file_exists(self):
        self.gc_pathhandler._maybe_make_directory('./tmp/')
        remote_path = '/'.join([self.gc_default_path, 'path/uploaded.txt'])
        local_path = './tmp/test_upload.txt'
        with open(local_path, 'w') as file:
            file.write('Local file to test uploading')
        isUploaded = self.gc_pathhandler._copy_from_local(local_path, remote_path)
        self.assertTrue(isUploaded)
        read = self.read_remote_file(remote_path, 'r')
        self.assertEqual(read, 'Local file to test uploading')
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_copy_from_local_file_doesnt_exist(self):
        local_path = '/file/that/doesnt/exist.txt'
        remote_path = '/'.join([self.gc_default_path, 'doesnt/exist.txt'])
        isUploaded = self.gc_pathhandler._copy_from_local(local_path, remote_path)
        self.assertFalse(isUploaded)
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_copy_remote_file_exists(self):
        remote_source = '/'.join([self.gc_default_path, 'path/uploaded.txt'])
        remote_destination = '/'.join([self.gc_default_path, 'path/uploaded-copy.txt'])
        isCopied = self.gc_pathhandler._copy(remote_source, remote_destination)
        self.assertTrue(isCopied)
        self.assertTrue(self.gc_pathhandler._exists(remote_destination))
        read = self.read_remote_file(remote_destination, 'r')
        self.assertEqual(read, 'Local file to test uploading')
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_copy_remote_file_doesnt_exist(self):
        remote_source = '/'.join([self.gc_default_path, 'doesnt/exist.txt'])
        remote_destination = '/'.join([self.gc_default_path, 'doesnt/exist-copy.txt'])
        isCopied = self.gc_pathhandler._copy(remote_source, remote_destination)
        self.assertFalse(isCopied)
        self.assertFalse(self.gc_pathhandler._exists(remote_destination))
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_get_local_path_remote_file_exists(self):
        remote_path = '/'.join([self.gc_default_path, 'path/uploaded.txt'])
        cache_path = self.gc_pathhandler._get_local_path(remote_path)
        self.assertEqual(cache_path, './tmp/test/path/uploaded.txt')
        with open(cache_path) as file:
            read = file.read()
        self.assertEqual(read, 'Local file to test uploading')
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_get_local_path_remote_file_doesnt_exists(self):
        remote_path = '/'.join([self.gc_default_path, 'will/exist.txt'])
        cache_path = self.gc_pathhandler._get_local_path(remote_path)
        self.assertEqual(cache_path, './tmp/test/will/exist.txt')
        self.assertTrue(os.path.exists(self.gc_pathhandler._get_local_cache_directory(remote_path)))
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_rm_when_remote_file_exists(self):
        remote_path = '/'.join([self.gc_default_path, 'path/uploaded-copy.txt'])
        self.assertTrue(self.gc_pathhandler._exists(remote_path))
        self.gc_pathhandler._rm(remote_path)
        self.assertFalse(self.gc_pathhandler._exists(remote_path))
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_rm_when_remote_file_doesnt_exist(self):
        remote_path = '/'.join([self.gc_default_path, 'doesnt/exist.txt'])
        self.assertFalse(self.gc_pathhandler._exists(remote_path))
        self.gc_pathhandler._rm(remote_path)
    @unittest.skipIf(not gc_auth, skip_gc_auth_required_tests_message)
    def test_rm_when_remote_path_is_directory(self):
        remote_path = '/'.join([self.gc_default_path, ''])
        self.assertTrue(self.gc_pathhandler._exists(remote_path))
        self.gc_pathhandler._rm(remote_path)
        self.assertTrue(self.gc_pathhandler._exists(remote_path))
    # ====>
