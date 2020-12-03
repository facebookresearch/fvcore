# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import base64
import errno
import logging
import os
import shutil
import tempfile
import traceback
from collections import OrderedDict
from typing import IO, Any, Callable, Dict, List, MutableMapping, Optional, Union
from urllib.parse import urlparse

import portalocker  # type: ignore
from fvcore.common.download import download


__all__ = ["LazyPath", "PathManager", "get_cache_dir", "file_lock"]


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
    try:
        PathManager.mkdirs(cache_dir)
        assert os.access(cache_dir, os.W_OK)
    except (OSError, AssertionError):
        tmp_dir = os.path.join(tempfile.gettempdir(), "fvcore_cache")
        logger = logging.getLogger(__name__)
        logger.warning(f"{cache_dir} is not accessible! Using {tmp_dir} instead!")
        cache_dir = tmp_dir
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
    return portalocker.Lock(path + ".lock", timeout=3600)  # type: ignore


class LazyPath(os.PathLike):
    """
    A path that's lazily evaluated when it's used.

    Users should be careful to not use it like a str, because
    it behaves differently from a str.
    Path manipulation functions in Python such as `os.path.*` all accept
    PathLike objects already.

    It can be materialized to a str using `os.fspath`.
    """

    def __init__(self, func: Callable[[], str]) -> None:
        """
        Args:
            func: a function that takes no arguments and returns the
                actual path as a str. It will be called at most once.
        """
        self._func = func
        self._value: Optional[str] = None

    def _get_value(self) -> str:
        if self._value is None:
            self._value = self._func()
        return self._value  # pyre-ignore

    def __fspath__(self) -> str:
        return self._get_value()

    # behave more like a str after evaluated
    def __getattr__(self, name: str):  # type: ignore
        if self._value is None:
            raise AttributeError(f"Uninitialized LazyPath has no attribute: {name}.")
        return getattr(self._value, name)

    def __getitem__(self, key):  # type: ignore
        if self._value is None:
            raise TypeError("Uninitialized LazyPath is not subscriptable.")
        return self._value[key]  # type: ignore

    def __str__(self) -> str:
        if self._value is not None:
            return self._value  # type: ignore
        else:
            return super().__str__()


class PathHandler:
    """
    PathHandler is a base class that defines common I/O functionality for a URI
    protocol. It routes I/O for a generic URI which may look like "protocol://*"
    or a canonical filepath "/foo/bar/baz".
    """

    _strict_kwargs_check = True

    def _check_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """
        Checks if the given arguments are empty. Throws a ValueError if strict
        kwargs checking is enabled and args are non-empty. If strict kwargs
        checking is disabled, only a warning is logged.

        Args:
            kwargs (Dict[str, Any])
        """
        if self._strict_kwargs_check:
            if len(kwargs) > 0:
                raise ValueError("Unused arguments: {}".format(kwargs))
        else:
            logger = logging.getLogger(__name__)
            for k, v in kwargs.items():
                logger.warning("[PathManager] {}={} argument ignored".format(k, v))

    def _get_supported_prefixes(self) -> List[str]:
        """
        Returns:
            List[str]: the list of URI prefixes this PathHandler can support
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def _copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def _exists(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        raise NotImplementedError()

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        raise NotImplementedError()

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def _rm(self, path: str, **kwargs: Any) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        raise NotImplementedError()

    def _symlink(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """
        Symlink the src_path to the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler to symlink from
            dst_path (str): A URI supported by this PathHandler to symlink to
        """
        raise NotImplementedError()


class NativePathHandler(PathHandler):
    """
    Handles paths that can be accessed using Python native system calls. This
    handler uses `open()` and `os.*` calls on the given path.
    """

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        self._check_kwargs(kwargs)
        return os.fspath(path)

    def _copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
        self._check_kwargs(kwargs)
        assert self._copy(
            src_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
        )

    def _open(
        self,
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        closefd: bool = True,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        opener: Optional[Callable] = None,
        **kwargs: Any,
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
            encoding (Optional[str]): the name of the encoding used to decode or
                encode the file. This should only be used in text mode.
            errors (Optional[str]): an optional string that specifies how encoding
                and decoding errors are to be handled. This cannot be used in binary
                mode.
            newline (Optional[str]): controls how universal newlines mode works
                (it only applies to text mode). It can be None, '', '\n', '\r',
                and '\r\n'.
            closefd (bool): If closefd is False and a file descriptor rather than
                a filename was given, the underlying file descriptor will be kept
                open when the file is closed. If a filename is given closefd must
                be True (the default) otherwise an error will be raised.
            opener (Optional[Callable]): A custom opener can be used by passing
                a callable as opener. The underlying file descriptor for the file
                object is then obtained by calling opener with (file, flags).
                opener must return an open file descriptor (passing os.open as opener
                results in functionality similar to passing None).

            See https://docs.python.org/3/library/functions.html#open for details.

        Returns:
            file: a file-like object.
        """
        self._check_kwargs(kwargs)
        return open(  # type: ignore
            path,
            mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
            opener=opener,
        )

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
        self._check_kwargs(kwargs)

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

    def _symlink(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """
        Creates a symlink to the src_path at the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler
            dst_path (str): A URI supported by this PathHandler

        Returns:
            status (bool): True on success
        """
        self._check_kwargs(kwargs)
        logger = logging.getLogger(__name__)
        if not os.path.exists(src_path):
            logger.error("Source path {} does not exist".format(src_path))
            return False
        if os.path.exists(dst_path):
            logger.error("Destination path {} already exists.".format(dst_path))
            return False
        try:
            os.symlink(src_path, dst_path)
            return True
        except Exception as e:
            logger.error("Error in symlink - {}".format(str(e)))
            return False

    def _exists(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.exists(path)

    def _isfile(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.isfile(path)

    def _isdir(self, path: str, **kwargs: Any) -> bool:
        self._check_kwargs(kwargs)
        return os.path.isdir(path)

    def _ls(self, path: str, **kwargs: Any) -> List[str]:
        self._check_kwargs(kwargs)
        return os.listdir(path)

    def _mkdirs(self, path: str, **kwargs: Any) -> None:
        self._check_kwargs(kwargs)
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            # EEXIST it can still happen if multiple processes are creating the dir
            if e.errno != errno.EEXIST:
                raise

    def _rm(self, path: str, **kwargs: Any) -> None:
        self._check_kwargs(kwargs)
        os.remove(path)


class HTTPURLHandler(PathHandler):
    """
    Download URLs and cache them to disk.
    """

    def __init__(self) -> None:
        self.cache_map: Dict[str, str] = {}

    def _get_supported_prefixes(self) -> List[str]:
        return ["http://", "https://", "ftp://"]

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        """
        This implementation downloads the remote resource and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        self._check_kwargs(kwargs)
        if path not in self.cache_map or not os.path.exists(self.cache_map[path]):
            logger = logging.getLogger(__name__)
            parsed_url = urlparse(path)
            dirname = os.path.join(
                get_cache_dir(), os.path.dirname(parsed_url.path.lstrip("/"))
            )
            filename = path.split("/")[-1]
            cached = os.path.join(dirname, filename)
            with file_lock(cached):
                if not os.path.isfile(cached):
                    cached = download(path, dirname, filename=filename)
            logger.info("URL {} cached in {}".format(path, cached))
            self.cache_map[path] = cached
        return self.cache_map[path]

    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
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
        self._check_kwargs(kwargs)
        assert mode in ("r", "rb"), "{} does not support open with {} mode".format(
            self.__class__.__name__, mode
        )
        assert (
            buffering == -1
        ), f"{self.__class__.__name__} does not support the `buffering` argument"
        local_path = self._get_local_path(path)
        return open(local_path, mode)


class OneDrivePathHandler(PathHandler):
    """
    Map OneDrive (short) URLs to direct download links
    """

    ONE_DRIVE_PREFIX = "https://1drv.ms/u/s!"

    def create_one_drive_direct_download(self, one_drive_url: str) -> str:
        """
        Converts a short OneDrive URI into a download link that can be used with wget

        Args:
            one_drive_url (str): A OneDrive URI supported by this PathHandler

        Returns:
            result_url (str): A direct download URI for the file
        """
        data_b64 = base64.b64encode(bytes(one_drive_url, "utf-8"))
        data_b64_string = (
            data_b64.decode("utf-8").replace("/", "_").replace("+", "-").rstrip("=")
        )
        result_url = (
            f"https://api.onedrive.com/v1.0/shares/u!{data_b64_string}/root/content"
        )
        return result_url

    def _get_supported_prefixes(self) -> List[str]:
        return [self.ONE_DRIVE_PREFIX]

    def _open(
        self, path: str, mode: str = "r", buffering: int = -1, **kwargs: Any
    ) -> Union[IO[str], IO[bytes]]:
        """
        Open a remote OneDrive path. The resource is first downloaded and cached
        locally.

        Args:
            path (str): A OneDrive URI supported by this PathHandler
            mode (str): Specifies the mode in which the file is opened. It defaults
                to 'r'.
            buffering (int): Not used for this PathHandler.

        Returns:
            file: a file-like object.
        """
        return PathManager.open(self._get_local_path(path), mode, **kwargs)

    def _get_local_path(self, path: str, **kwargs: Any) -> str:
        """
        This implementation downloads the remote resource and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        logger = logging.getLogger(__name__)
        direct_url = self.create_one_drive_direct_download(path)

        logger.info(f"URL {path} mapped to direct download link {direct_url}")

        return PathManager.get_local_path(os.fspath(direct_url), **kwargs)


# NOTE: this class should be renamed back to PathManager when it is moved to a new library
class PathManagerBase:
    """
    A class for users to open generic paths or translate generic paths to file names.

    path_manager.method(path) will do the following:
    1. Find a handler by checking the prefixes in `self._path_handlers`.
    2. Call handler.method(path) on the handler that's found
    """

    def __init__(self) -> None:
        self._path_handlers: MutableMapping[str, PathHandler] = OrderedDict()
        """
        Dict from path prefix to handler.
        """

        self._native_path_handler: PathHandler = NativePathHandler()
        """
        A NativePathHandler that works on posix paths. This is used as the fallback.
        """

    def __get_path_handler(self, path: Union[str, os.PathLike]) -> PathHandler:
        """
        Finds a PathHandler that supports the given path. Falls back to the native
        PathHandler if no other handler is found.

        Args:
            path (str or os.PathLike): URI path to resource

        Returns:
            handler (PathHandler)
        """
        path = os.fspath(path)  # pyre-ignore
        for p in self._path_handlers.keys():
            if path.startswith(p):
                return self._path_handlers[p]
        return self._native_path_handler

    def open(
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
        return self.__get_path_handler(path)._open(  # type: ignore
            path, mode, buffering=buffering, **kwargs
        )

    def copy(
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

        # Copying across handlers is not supported.
        assert self.__get_path_handler(  # type: ignore
            src_path
        ) == self.__get_path_handler(dst_path)
        return self.__get_path_handler(src_path)._copy(
            src_path, dst_path, overwrite, **kwargs
        )

    def get_local_path(self, path: str, **kwargs: Any) -> str:
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
        path = os.fspath(path)
        return self.__get_path_handler(path)._get_local_path(  # type: ignore
            path, **kwargs
        )

    def copy_from_local(
        self, local_path: str, dst_path: str, overwrite: bool = False, **kwargs: Any
    ) -> None:
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
        assert os.path.exists(local_path)
        return self.__get_path_handler(dst_path)._copy_from_local(
            local_path=local_path, dst_path=dst_path, overwrite=overwrite, **kwargs
        )

    def exists(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there is a resource at the given URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path exists
        """
        return self.__get_path_handler(path)._exists(path, **kwargs)  # type: ignore

    def isfile(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if there the resource at the given URI is a file.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a file
        """
        return self.__get_path_handler(path)._isfile(path, **kwargs)  # type: ignore

    def isdir(self, path: str, **kwargs: Any) -> bool:
        """
        Checks if the resource at the given URI is a directory.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            bool: true if the path is a directory
        """
        return self.__get_path_handler(path)._isdir(path, **kwargs)  # type: ignore

    def ls(self, path: str, **kwargs: Any) -> List[str]:
        """
        List the contents of the directory at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler

        Returns:
            List[str]: list of contents in given path
        """
        return self.__get_path_handler(path)._ls(path, **kwargs)

    def mkdirs(self, path: str, **kwargs: Any) -> None:
        """
        Recursive directory creation function. Like mkdir(), but makes all
        intermediate-level directories needed to contain the leaf directory.
        Similar to the native `os.makedirs`.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return self.__get_path_handler(path)._mkdirs(path, **kwargs)  # type: ignore

    def rm(self, path: str, **kwargs: Any) -> None:
        """
        Remove the file (not directory) at the provided URI.

        Args:
            path (str): A URI supported by this PathHandler
        """
        return self.__get_path_handler(path)._rm(path, **kwargs)  # type: ignore

    def symlink(self, src_path: str, dst_path: str, **kwargs: Any) -> bool:
        """Symlink the src_path to the dst_path

        Args:
            src_path (str): A URI supported by this PathHandler to symlink from
            dst_path (str): A URI supported by this PathHandler to symlink to
        """
        # Copying across handlers is not supported.
        assert self.__get_path_handler(  # type: ignore
            src_path
        ) == self.__get_path_handler(dst_path)
        return self.__get_path_handler(src_path)._symlink(src_path, dst_path, **kwargs)

    def register_handler(
        self, handler: PathHandler, allow_override: bool = False
    ) -> None:
        """
        Register a path handler associated with `handler._get_supported_prefixes`
        URI prefixes.

        Args:
            handler (PathHandler)
            allow_override (bool): allow overriding existing handler for prefix
        """
        logger = logging.getLogger(__name__)
        assert isinstance(handler, PathHandler), handler
        for prefix in handler._get_supported_prefixes():
            if prefix not in self._path_handlers:
                self._path_handlers[prefix] = handler
                continue

            old_handler_type = type(self._path_handlers[prefix])
            if allow_override:
                # if using the global PathManager, show the warnings
                if self == PathManager:
                    logger.warning(
                        f"[PathManager] Attempting to register prefix '{prefix}' from "
                        "the following call stack:\n"
                        + "".join(traceback.format_stack(limit=5))
                        # show the most recent callstack
                    )
                    logger.warning(
                        f"[PathManager] Prefix '{prefix}' is already registered "
                        f"by {old_handler_type}. We will override the old handler. "
                        "To avoid such conflicts, create a project-specific PathManager "
                        "instead."
                    )
                self._path_handlers[prefix] = handler
            else:
                raise KeyError(
                    f"[PathManager] Prefix '{prefix}' already registered by {old_handler_type}!"
                )

        # Sort path handlers in reverse order so longer prefixes take priority,
        # eg: http://foo/bar before http://foo
        self._path_handlers = OrderedDict(
            sorted(self._path_handlers.items(), key=lambda t: t[0], reverse=True)
        )

    def set_strict_kwargs_checking(self, enable: bool) -> None:
        """
        Toggles strict kwargs checking. If enabled, a ValueError is thrown if any
        unused parameters are passed to a PathHandler function. If disabled, only
        a warning is given.

        With a centralized file API, there's a tradeoff of convenience and
        correctness delegating arguments to the proper I/O layers. An underlying
        `PathHandler` may support custom arguments which should not be statically
        exposed on the `PathManager` function. For example, a custom `HTTPURLHandler`
        may want to expose a `cache_timeout` argument for `open()` which specifies
        how old a locally cached resource can be before it's refetched from the
        remote server. This argument would not make sense for a `NativePathHandler`.
        If strict kwargs checking is disabled, `cache_timeout` can be passed to
        `PathManager.open` which will forward the arguments to the underlying
        handler. By default, checking is enabled since it is innately unsafe:
        multiple `PathHandler`s could reuse arguments with different semantic
        meanings or types.

        Args:
            enable (bool)
        """
        self._native_path_handler._strict_kwargs_check = enable
        for handler in self._path_handlers.values():
            handler._strict_kwargs_check = enable


PathManager = PathManagerBase()
"""
A global PathManager.

Any sufficiently complicated/important project should create their own
PathManager instead of using the global PathManager, to avoid conflicts
when multiple projects have conflicting PathHandlers.

History: at first, PathManager is part of detectron2 *only*, and therefore
does not consider cross-projects conflict issues. It is later used by more
projects and moved to fvcore to faciliate more use across projects and lead
to some conflicts.
Now the class `PathManagerBase` is added to help create per-project path manager,
and this global is still named "PathManager" to keep backward compatibility.
"""

PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())
