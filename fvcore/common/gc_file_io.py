import io
import os
import threading
import types

from google.cloud import storage
from typing import *

from fvcore.common.file_io import PathHandler, PathManager

mutex = threading.Lock()

class GoogleCloudHandler(PathHandler):

    def _get_supported_prefixes(self) -> List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        return ["gs://"]

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.
        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, the cache stays on filesystem
        (under `file_io.get_cache_dir()`) and will be used by a different run.
        Therefore this function is meant to be used with read-only resources.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            local_path (str): a file path which exists on the local file system
        """
        self._cache_remote_file(path)
        return get_local_cache_path(path)

    def _copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a local file to the specified URI.
        If the URI is another local path, this should be functionally identical
        to copy.
        Args:
            local_path (str): a file path which exists on the local file system
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing URI
        Returns:
            status (bool): True on success
        """
        try:
            self._upload_file(dst_path, local_path)
            return True
        except:
            return False

    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a stream to a URI, similar to the built-in `open`.
        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy depends on the
                underlying I/O implementation.
        Returns:
            file: a file-like object.
        """
        self._cache_remote_file(path)
        return self._open_local_copy(path, mode) 

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> bool:
        """
        Copies a source path to a destination path.
        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file
        Returns:
            status (bool): True on success
        """
        try: self._cache_remote_file(src_path)
        except: return False
        local_path = get_local_cache_path(src_path)
        return self._copy_from_local(local_path, dst_path)

    def _exists(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path exists
        """
        return self._get_blob(path).exists()
        

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a file.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path is a file
        """
        
        return '.' in path.split('/')[-1]

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a directory.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            bool: true if the path is a directory
        """
        return '/' == path[-1]

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        """
        List the contents of the directory at the provided URI.
        Args:
            path (str): A URI supported by this PathHandler
        Returns:
            List[str]: list of contents in given path
        """
        raise NotImplementedError()

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.
        Args:
            path (str): A URI supported by this PathHandler
        """
        # GCS does this automatically
        pass

    def _rm(self, path: str, **kwargs: Any) -> None:
        """
        Remove the file (not directory) at the provided URI.
        Args:
            path (str): A URI supported by this PathHandler
        """
        if not self._exists(path): return
        if self._isdir(path): return
        self._delete_remote_resource(path)    
   
    def _get_gc_bucket(self, path:str) -> storage.Bucket:
        if not hasattr(self, '_gc_client'):
            self._create_gc_client(path)
        gc_bucket_name = extract_gc_bucket_name(path)
        return self._gc_client.get_bucket(gc_bucket_name)

    def _create_gc_client(self, path:str):
        namespace = extract_gc_namespace(path)
        gc_client = storage.Client(project=namespace)
        setattr(self, '_gc_client', gc_client)
    
    def _get_blob(self, path:str) -> storage.Blob:
        gc_bucket = self._get_gc_bucket(path)
        return gc_bucket.blob(extract_blob_path(path))
    
    def _cache_blob(self, local_path:str, gc_blob:storage.Blob):
        if not gc_blob.exists(): return
        with open(local_path, 'wb') as file:
            gc_blob.download_to_file(file)

    def _upload_file(self, destination_path:str, local_path:str):
        gc_blob = self._get_blob(destination_path)
        with open(local_path, 'r') as file:
            gc_blob.upload_from_file(file)

    def _cache_remote_file(self, remote_path:str):
        local_path = get_local_cache_path(remote_path)
        local_directory = get_local_cache_directory(remote_path)
        maybe_make_directory(local_directory)
        gc_blob = self._get_blob(remote_path)
        self._cache_blob(local_path, gc_blob)
        
    def _open_local_copy(self, path: str, mode: str) -> Union[IO[str], IO[bytes]]:
        local_path = get_local_cache_path(path)
        gc_blob = self._get_blob(path)
        file = open(local_path, mode)
        if 'w' in mode:
            decorate_file_with_gc_methods(file, gc_blob)
        return file
    
    def _delete_remote_resource(self, path):
        self._get_blob(path).delete()


def close_and_upload(self):
    mode = self.mode
    name = self.name
    self._close()
    with open(name, mode.replace('w', 'r')) as file_to_upload:
        self._gc_blob.upload_from_file(file_to_upload)

def decorate_file_with_gc_methods(
    file: Union[IO[str], IO[bytes]],
    gc_blob: storage.Blob
):    
    setattr(file, '_gc_blob', gc_blob)
    setattr(file, '_close', file.close)
    file.close = types.MethodType(close_and_upload, file)

def maybe_make_directory(path:str) -> bool:
    is_made = False

    mutex.acquire()
    if not os.path.exists(path):
        os.makedirs(path)
        is_made = True
    mutex.release()
    
    return is_made

def extract_gc_namespace(path:str) -> str:
    return extract_gc_bucket_name(path).replace("-data", "")
def extract_gc_bucket_name(path:str) -> str:
    return remove_file_system(path).split("/")[0]
def remove_file_system(path:str) -> str:
    return path.replace("gs://", "")
def remove_bucket_name(path:str) -> str:
    return path.replace(extract_gc_bucket_name(path)+"/", "")

def extract_blob_path(path:str) -> str:
    return remove_file_system(remove_bucket_name(path))    

def get_local_cache_path(path:str) -> str:
    path = extract_blob_path(path)
    return '/'.join(['.','tmp', path])

def get_local_cache_directory(path:str) -> str:
    path = get_local_cache_path(path)
    return path.replace(path.split('/')[-1], '')