"""Find the members of one dump-format clip, whether it is a directory or a ``<name>.zip``.

The extra training pool (``--extra_pool``) holds clips in the layout 4I4's clip writer
(``triton/4rc/multiview/clip_writer.py``) produces for the Syn4D converter: three members side by
side, in a clip directory or at the root of ``<name>.zip``::

    meta.npz        seq_name, query_points, traj3d_world, visibility, intrs, extrs, depth0,
                    view_ids, track_upscaling_factor
    depth_full.npz  key "depth", (V,T,1,H,W) metres, 0 where invalid
    frames.zip      view_<v>/<t:04d>.png, v the position on the view axis

This module owns that layout the way the vendored ``scene_archive`` owns MV-Kubric's, and like it
imports no torch, so ``--build_pool_index`` reads a clip's view count through it.  What the arrays
must satisfy is :func:`arc.training.dumped_kubric.read_clip`'s question; this module finds the
members, and turns a member that cannot be read into a :class:`ClipFormatError` naming the clip.

**Zip clips are read into memory.**  An npz read through a zip member re-reads the member on every
backwards seek (``ZipExtFile`` rewinds to get there), and ``frames.zip`` nested in a clip zip would
do the same once per frame; the clip's members are stored, so reading one is a copy, and the hub
measured a whole clip -- read and every frame decoded -- at 0.65-0.86 s on one core.  A directory
clip is read in place, its ``frames.zip`` opened from disk so only the planned frames are read.
"""

from __future__ import annotations

import contextlib
import io
import os
import struct
import zipfile
import zlib
from typing import Iterator

import numpy as np

from arc.training.scene_archive import ARCHIVE_SUFFIX

META_NAME = "meta.npz"
DEPTH_NAME = "depth_full.npz"
DEPTH_KEY = "depth"
FRAMES_NAME = "frames.zip"
CLIP_MEMBERS = (META_NAME, DEPTH_NAME, FRAMES_NAME)
META_KEYS = (
    "seq_name",
    "query_points",
    "traj3d_world",
    "visibility",
    "intrs",
    "extrs",
    "depth0",
    "view_ids",
    "track_upscaling_factor",
)
# Recorded in a clip pool's pool_index.json, and absent from an MV-Kubric pool's, so a pool given
# to the wrong flag is refused when the index is read rather than one load at a time.
CLIP_POOL_KIND = "dump_clips"

# What a damaged member raises on the way in: a missing or truncated file, a corrupt archive or
# compressed stream, a missing npz key, an npz that would need pickle, and -- from PIL -- an image
# it cannot identify (an OSError) or a broken PNG chunk (SyntaxError or struct.error).
READ_ERRORS = (
    OSError,
    EOFError,
    KeyError,
    ValueError,
    SyntaxError,
    struct.error,
    zipfile.BadZipFile,
    zlib.error,
)


class ClipFormatError(ValueError):
    """A clip that does not meet the dump-format contract; the message names the clip."""


@contextlib.contextmanager
def reading(label: str, member: str) -> Iterator[None]:
    """Turn a failed read -- or a contract check that raised ValueError -- into a ClipFormatError.

    Scoped to one member, so the message says which of the three files is wrong.
    """

    try:
        yield
    except ClipFormatError:
        raise
    except READ_ERRORS as error:
        raise ClipFormatError(
            f"{label}: {member}: {type(error).__name__}: {error}"
        ) from error


class Clip:
    """One open clip: its three members, whichever layout it is on disk.

    Exactly one of ``directory`` (a clip directory, read in place) and ``read_member`` (a
    zip clip's member bytes, read once) is given.
    """

    def __init__(self, label: str, *, directory: str | None = None, read_member=None):
        if (directory is None) == (read_member is None):
            raise TypeError("a Clip reads from a directory or from member bytes, not both")
        self.label = label
        self._directory = directory
        self._read_member = read_member
        self._frames: zipfile.ZipFile | None = None

    def load_npz(self, member: str):
        """``np.load`` of one npz member, pickle refused; use it as a context manager."""

        with reading(self.label, member):
            if self._directory is not None:
                # Straight from the path, as scene_archive's DirectoryScene loads its npz.
                return np.load(os.path.join(self._directory, member), allow_pickle=False)
            return np.load(io.BytesIO(self._read_member(member)), allow_pickle=False)

    def frames(self) -> zipfile.ZipFile:
        """``frames.zip``, opened once per clip and closed with it."""

        if self._frames is None:
            with reading(self.label, FRAMES_NAME):
                self._frames = zipfile.ZipFile(
                    os.path.join(self._directory, FRAMES_NAME)
                    if self._directory is not None
                    else io.BytesIO(self._read_member(FRAMES_NAME))
                )
        return self._frames

    def close(self) -> None:
        if self._frames is not None:
            self._frames.close()
            self._frames = None


def _reject_wrapped_archive(label: str, names: set[str]) -> None:
    """Refuse ``zip -r <name>.zip <name>``, which nests the members under an extra directory.

    The same refusal ``scene_archive.ZipScene`` makes for an MV-Kubric scene: reporting the
    members as missing would hide a whole pool packed from the wrong directory.
    """

    if META_NAME in names:
        return
    nested = sorted(
        name.split("/", 1)[0] + "/"
        for name in names
        if "/" in name and name.split("/", 1)[1] == META_NAME
    )
    if nested:
        raise ClipFormatError(
            f"{label}: members are nested under {nested[0]!r} instead of being at the "
            "archive root. Pack from inside the clip directory, not from its parent"
        )


@contextlib.contextmanager
def open_clip(path) -> Iterator[Clip]:
    """Open the clip at ``path``: a directory, or a ``<name>.zip`` holding the members at its root.

    Every member must be present; one that is not, an unreadable archive and a wrapped one are
    refused as :class:`ClipFormatError`.  Hold it open for one clip's load.
    """

    path = os.fspath(path)
    if os.path.isdir(path):
        missing = [
            member
            for member in CLIP_MEMBERS
            if not os.path.isfile(os.path.join(path, member))
        ]
        if missing:
            raise ClipFormatError(f"{path}: missing {missing}")
        clip = Clip(path, directory=path)
        try:
            yield clip
        finally:
            clip.close()
        return
    if not (path.endswith(ARCHIVE_SUFFIX) and os.path.isfile(path)):
        raise ClipFormatError(
            f"no clip at {path!r}: neither a directory nor a <name>{ARCHIVE_SUFFIX} archive"
        )
    with reading(path, "archive"):
        archive = zipfile.ZipFile(path)
    with archive:
        names = set(archive.namelist())
        _reject_wrapped_archive(path, names)
        missing = [member for member in CLIP_MEMBERS if member not in names]
        if missing:
            raise ClipFormatError(f"{path}: missing {missing}")
        members: dict[str, bytes] = {}

        def read_member(member: str) -> bytes:
            if member not in members:
                members[member] = archive.read(member)
            return members[member]

        clip = Clip(path, read_member=read_member)
        try:
            yield clip
        finally:
            clip.close()


def checked_view_ids(view_ids, label: str) -> np.ndarray:
    """``view_ids`` as int64, refused unless a non-empty 1-D set of distinct non-negative ints.

    They are the source camera ids, and may skip -- the converter drops a camera that sees too
    little within the depth cap -- while every array stays indexed by position on the view axis.
    """

    array = np.asarray(view_ids)
    if array.ndim != 1 or array.shape[0] < 1:
        raise ClipFormatError(
            f"{label}: view_ids must be a non-empty (V,) array, got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ClipFormatError(
            f"{label}: view_ids must be integers, got {array.dtype}"
        )
    ids = array.astype(np.int64)
    if np.any(ids < 0):
        raise ClipFormatError(f"{label}: view_ids must be non-negative, got {ids.tolist()}")
    if len(set(ids.tolist())) != ids.shape[0]:
        raise ClipFormatError(f"{label}: view_ids must be distinct, got {ids.tolist()}")
    return ids


def clip_view_ids(path) -> np.ndarray:
    """One clip's checked ``view_ids``, the population of its view draw; reads no other array."""

    with open_clip(path) as clip:
        with reading(clip.label, META_NAME):
            with clip.load_npz(META_NAME) as meta:
                if "view_ids" not in meta.files:
                    raise ClipFormatError(f"{clip.label}: {META_NAME} lacks view_ids")
                view_ids = meta["view_ids"]
        return checked_view_ids(view_ids, clip.label)
