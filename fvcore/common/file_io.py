# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
import tempfile
from typing import Optional

from iopath.common.file_io import (  # noqa, unused import required by some deps
    file_lock,
    HTTPURLHandler,
    LazyPath,
    NativePathHandler,
    OneDrivePathHandler,
    PathHandler,
    PathManager as PathManagerBase,
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
