
import inspect
import io
import os
import shutil
import unittest
from typing import IO, Union

from fvcore.common.file_io import close_and_upload, GoogleCloudHandler
from google.cloud import storage


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

if __name__ == '__main__':
    unittest.main()