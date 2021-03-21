# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
import tempfile
import traceback
from collections import OrderedDict
from typing import IO, Any, List, MutableMapping, Optional, Union

from iopath.common.file_io import (
    HTTPURLHandler,
    LazyPath,
    NativePathHandler,
    OneDrivePathHandler,
    PathHandler,
    file_lock,
)


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
        logger = logging.getLogger(__name__)
        logger.warning(
            "** fvcore version of PathManager will be deprecated soon. **\n"
            "** Please migrate to the version in iopath repo. **\n"
            "https://github.com/facebookresearch/iopath \n"
        )

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

    def get_local_path(self, path: str, force: bool = False, **kwargs: Any) -> str:
        """
        Get a filepath which is compatible with native Python I/O such as `open`
        and `os.path`.

        If URI points to a remote resource, this function may download and cache
        the resource to local disk.

        Args:
            path (str): A URI supported by this PathHandler
            force(bool): Forces a download from backend if set to True.

        Returns:
            local_path (str): a file path which exists on the local file system
        """
        path = os.fspath(path)
        return self.__get_path_handler(path)._get_local_path(  # type: ignore
            path, force=force, **kwargs
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
