#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import errno
import logging
import os
import shutil
from collections import OrderedDict
from typing import IO, Any, Dict, List, MutableMapping, Optional, Union
from urllib.parse import urlparse

from fvcore.common.download import download

import portalocker  # type: ignore

__all__ = ["PathManager", "get_cache_dir", "file_lock"]


def get_cache_dir(cache_dir: Optional[str] = None) -> str:
    """
    Returns a default directory to cache static files
    (usually downloaded from Internet), if None is provided.

    Args:
        cache_dir (None or str): if not None, will be returned as is.
            If None, returns the default cache directory as:

        1) $FVCORE_CACHE, if set
        2) otherwise ~/.torch/fvcore_cache
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser(
            os.getenv("FVCORE_CACHE", "~/.torch/fvcore_cache")
        )
    return cache_dir


def file_lock(path: str):  # type: ignore
    """
    A file lock. Once entered, it is guaranteed that no one else holds the
    same lock. Others trying to enter the lock will block for 30 minutes and
    raise an exception.

    This is useful to make sure workers don't cache files to the same location.

    Args:
        path (str): a path to be locked. This function will create a lock named
            `path + ".lock"`

    Examples:

        filename = "/path/to/file"
        with file_lock(filename):
            if not os.path.isfile(filename):
                do_create_file()
    """
    dirname = os.path.dirname(path)
    try:
        os.makedirs(dirname, exist_ok=True)
    except OSError:
        # makedir is not atomic. Exceptions can happen when multiple workers try
        # to create the same dir, despite exist_ok=True.
        # When this happens, we assume the dir is created and proceed to creating
        # the lock. If failed to create the directory, the next line will raise
        # exceptions.
        pass
    return portalocker.Lock(path + ".lock", timeout=1800)  # type: ignore


class PathHandler:
    """
    PathHandler is a base class that defines common I/O functionality for a URI
    protocol. It routes I/O for a generic URI which may look like "protocol://*"
    or a canonical filepath "/foo/bar/baz".
    """

    def _get_supported_prefixes(self) -> List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        raise NotImplementedError()

    def _get_local_path(self, path: str) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk. In this case, this function is meant to be
        used with read-only resources.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        raise NotImplementedError()

    def _open(
        self, path: str, mode: str = "r", buffering: int = -1
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
        raise NotImplementedError()

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False
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
        raise NotImplementedError()

    def _exists(self, path: str) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        raise NotImplementedError()

    def _isfile(self, path: str) -> bool:
        """
        Checks if the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        raise NotImplementedError()

    def _isdir(self, path: str) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        raise NotImplementedError()

    def _ls(self, path: str) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        raise NotImplementedError()

    def _mkdirs(self, path: str) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def _rm(self, path: str) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()


class NativePathHandler(PathHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.
    """

    def _get_local_path(self, path: str) -> str:
        return path

    def _open(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        **kwargs: Dict[str, Any],
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a path.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): An optional integer used to set the buffering policy.
                Pass 0 to switch buffering off and an integer >= 1 to indicate the
                size in bytes of a fixed-size chunk buffer. When no buffering
                argument is given, the default buffering policy works as follows:
                    * Binary files are buffered in fixed-size chunks; the size of
                    the buffer is chosen using a heuristic trying to determine the
                    underlying device’s “block size” and falling back on
                    io.DEFAULT_BUFFER_SIZE. On many systems, the buffer will
                    typically be 4096 or 8192 bytes long.

        Returns:
            file: a file-like object.
        """
        return open(path, mode, buffering=buffering)

    def _copy(
        self, src_path: str, dst_path: str, overwrite: bool = False
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
        if os.path.exists(dst_path) and not overwrite:
            logger = logging.getLogger(__name__)
            logger.error("Destination file {} already exists.".format(dst_path))
            return False

        try:
            shutil.copyfile(src_path, dst_path)
            return True
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error("Error in file copy - {}".format(str(e)))
            return False

    def _exists(self, path: str) -> bool:
        return os.path.exists(path)

    def _isfile(self, path: str) -> bool:
        return os.path.isfile(path)

    def _isdir(self, path: str) -> bool:
        return os.path.isdir(path)

    def _ls(self, path: str) -> List[str]:
        return os.listdir(path)

    def _mkdirs(self, path: str) -> None:
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            # EEXIST it can still happen if multiple processes are creating the dir
            if e.errno != errno.EEXIST:
                raise

    def _rm(self, path: str) -> None:
        os.remove(path)


class HTTPURLHandler(PathHandler):
    """
    Download URLs and cache them to disk.
    """

    def __init__(self) -> None:
        self.cache_map: Dict[str, str] = {}

    def _get_supported_prefixes(self) -> List[str]:
        return ["http://", "https://", "ftp://"]

    def _get_local_path(self, path: str) -> str:
        """
        This implementation downloads the remote resource and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        if path not in self.cache_map or not os.path.exists(
            self.cache_map[path]
        ):
            logger = logging.getLogger(__name__)
            parsed_url = urlparse(path)
            dirname = os.path.join(
                get_cache_dir(), os.path.dirname(parsed_url.path.lstrip("/"))
            )
            filename = path.split("/")[-1]
            cached = os.path.join(dirname, filename)
            with file_lock(cached):
                if not os.path.isfile(cached):
                    logger.info("Downloading {} ...".format(path))
                    cached = download(path, dirname, filename=filename)
            logger.info("URL {} cached in {}".format(path, cached))
            self.cache_map[path] = cached
        return self.cache_map[path]

    def _open(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        **kwargs: Dict[str, Any],
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a remote HTTP path. The resource is first downloaded and cached
        locally.

        Args:
            path (str): A URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): Not used for this PathHandler.

        Returns:
            file: a file-like object.
        """
        assert mode in (
            "r",
            "rb",
        ), "{} does not support open with {} mode".format(
            self.__class__.__name__, mode
        )
        assert (
            buffering == -1
        ), f"{self.__class__.__name__} does not support the `buffering` argument"
        local_path = self._get_local_path(path)
        return open(local_path, mode)


class PathManager:
    """
    A class for users to open generic paths or translate generic paths to file names.
    """

    _PATH_HANDLERS: MutableMapping[str, PathHandler] = OrderedDict()
    _NATIVE_PATH_HANDLER = NativePathHandler()

    @staticmethod
    def __get_path_handler(path: str) -> PathHandler:
        """
        Finds a PathHandler that supports the given path. Falls back to the native
        PathHandler if no other handler is found.

        Args:
            path (str): URI path to resource

        Returns:
            handler (PathHandler)
        """
        for p in PathManager._PATH_HANDLERS.keys():
            if path.startswith(p):
                return PathManager._PATH_HANDLERS[p]
        return PathManager._NATIVE_PATH_HANDLER

    @staticmethod
    def open(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        **kwargs: Dict[str, Any],
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
        return PathManager.__get_path_handler(path)._open(  # type: ignore
            path, mode, buffering=buffering
        )

    @staticmethod
    def copy(src_path: str, dst_path: str, overwrite: bool = False) -> bool:
        """
        Copies a source path to a destination path.

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler
            overwrite (bool): Bool flag for forcing overwrite of existing file

        Returns:
            status (bool): True on success
        """

        # Copying across handlers is not supported.
        assert PathManager.__get_path_handler(  # type: ignore
            src_path
        ) == PathManager.__get_path_handler(dst_path)
        return PathManager.__get_path_handler(src_path)._copy(
            src_path, dst_path, overwrite
        )

    @staticmethod
    def get_local_path(path: str) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        return PathManager.__get_path_handler(  # type: ignore
            path
        )._get_local_path(path)

    @staticmethod
    def exists(path: str) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        return PathManager.__get_path_handler(path)._exists(  # type: ignore
            path
        )

    @staticmethod
    def isfile(path: str) -> bool:
        """
        Checks if there the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        return PathManager.__get_path_handler(path)._isfile(  # type: ignore
            path
        )

    @staticmethod
    def isdir(path: str) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        return PathManager.__get_path_handler(path)._isdir(path)  # type: ignore

    @staticmethod
    def ls(path: str) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        return PathManager.__get_path_handler(path)._ls(path)  # type: ignore

    @staticmethod
    def mkdirs(path: str) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return PathManager.__get_path_handler(path)._mkdirs(  # type: ignore
            path
        )

    @staticmethod
    def rm(path: str) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return PathManager.__get_path_handler(path)._rm(path)  # type: ignore

    @staticmethod
    def register_handler(handler: PathHandler) -> None:
        """
        Register a path handler associated with `handler._get_supported_prefixes`
        URI prefixes.

        Args:
            handler (PathHandler)
        """
        assert isinstance(handler, PathHandler), handler
        for prefix in handler._get_supported_prefixes():
            assert prefix not in PathManager._PATH_HANDLERS
            PathManager._PATH_HANDLERS[prefix] = handler

        # Sort path handlers in reverse order so longer prefixes take priority,
        # eg: http://foo/bar before http://foo
        PathManager._PATH_HANDLERS = OrderedDict(
            sorted(
                PathManager._PATH_HANDLERS.items(),
                key=lambda t: t[0],
                reverse=True,
            )
        )


PathManager.register_handler(HTTPURLHandler())
