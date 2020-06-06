# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import os
import shutil
from typing import Callable, List, Optional
from urllib import request


def dump_url_to_file(
    url: str, filepath: str, progress: bool = True, desc: str = None
) -> str:
    """
    Download a file from a given URL to a directory. If file exists, will
        overwrite the existing file.

    Args:
        url (str):
        filepath (str): the path to save the file.
            The directory is assumed to already exist.
        progress (bool): whether to use tqdm to draw a progress bar.
        desc (bool): desc to pass to tqdm if drawing a progress bar.

    Returns:
        filepath (str): the path to the downloaded file.
            This is always identical to the filepath argument.
    """
    if progress:
        import tqdm

        def hook(t: tqdm.tqdm) -> Callable[[int, int, Optional[int]], None]:
            last_b: List[int] = [0]

            def inner(b: int, bsize: int, tsize: Optional[int] = None) -> None:
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b[0]) * bsize)  # type: ignore
                last_b[0] = b

            return inner

        with tqdm.tqdm(  # type: ignore
            unit="B", unit_scale=True, miniters=1, desc=desc, leave=True
        ) as t:
            tmp, _ = request.urlretrieve(url, filename=filepath, reporthook=hook(t))

    else:
        tmp, _ = request.urlretrieve(url, filename=filepath)
    return tmp


def download(
    url: str, dir: str, *, filename: Optional[str] = None, progress: bool = True
) -> str:
    """
    Download a file from a given URL to a directory. If file exists, will not
        overwrite the existing file.

    Args:
        url (str):
        dir (str): the directory to download the file
        filename (str or None): the basename to save the file.
            Will use the name in the URL if not given.
        progress (bool): whether to use tqdm to draw a progress bar.

    Returns:
        str: the path to the downloaded file or the existing one.
    """
    os.makedirs(dir, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1]
        assert len(filename), "Cannot obtain filename from url {}".format(url)
    fpath = os.path.join(dir, filename)
    logger = logging.getLogger(__name__)

    if os.path.isfile(fpath):
        logger.info("File {} exists! Skipping download.".format(filename))
        return fpath

    tmp = fpath + ".tmp"  # download to a tmp file first, to be more atomic.
    try:
        logger.info("Downloading from {} ...".format(url))
        tmp = dump_url_to_file(url, filepath=tmp, progress=progress, desc=filename)
        statinfo = os.stat(tmp)
        size = statinfo.st_size
        if size == 0:
            raise IOError("Downloaded an empty file from {}!".format(url))
        # download to tmp first and move to fpath, to make this function more
        # atomic.
        shutil.move(tmp, fpath)
    except IOError:
        logger.error("Failed to download {}".format(url))
        raise
    finally:
        try:
            os.unlink(tmp)
        except IOError:
            pass

    logger.info("Successfully downloaded " + fpath + ". " + str(size) + " bytes.")
    return fpath
