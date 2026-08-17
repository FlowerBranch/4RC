# VENDORED, DO NOT EDIT. Copy of mvtracker/datasets/scene_archive.py, which owns the on-disk
# layout of an MV-Kubric scene and is stdlib+numpy only, so a consuming repo can vendor it rather
# than re-derive the layout.
#
#   upstream : mvtracker @ 4002795 ("fix: correct the manifest replay spec for packed pools
#              and refuse a multi-GPU manifest")
#   sha256   : 32416073a1b7db2c67eb7804695655d8e83196da4a65c503d0a4ef4faac49677
#
# The body below is byte-identical to upstream; only this header is local, so the hash above is
# taken over the file WITHOUT it, and `tests/test_scene_provider.py` recomputes it. Re-vendor on a
# change upstream, never patch here: the view-ordering rule in the docstring is subtle enough that
# two copies drifting apart would mis-index cameras with nothing downstream able to notice.
"""Read one MV-Kubric scene, whether it is a directory or a `<scene>.zip` archive.

Why archives exist here: the project quota caps file *count*, and a scene ships 784 files, so the
4970-scene split needs 3.9M against a ~1,048k cap and cannot be staged loose. Packing each scene into
one archive makes the split 4,970 files. Members mirror the directory tree exactly and are stored
uncompressed, so a read is a seek and a copy.

Layout is chosen by what is on disk, never by config: a benchmark set can stay loose while the
training pool is packed, in the same run.

View indices are positions, not names
-------------------------------------
The loader's view axis is built by appending views in numerically-sorted folder order, so every index
in ``views_to_return`` / ``novel_views`` is a **position in that ordered list**, never the N in
``view_N``. Both plausible shortcuts are silently wrong: ``f"view_{i}"`` breaks on a scene whose views
are ``view_0``, ``view_2``, ``view_5``; and lexicographic ordering puts ``view_10`` between ``view_1``
and ``view_2``, which is live for the 25-view configuration. Zero-padded names (``view_04``) also
exist in this tree. Nothing downstream catches a mix-up either -- the projection sanity check uses
each view's own K, E and 2D tracks, so it passes for any view.

Hence one rule, shared by both readers and implemented once in :func:`_ordered_view_prefixes`:

    take the distinct first path components that begin with ``view_`` AND are followed by a
    separator; order them by ``int(name.split("_")[-1])``; position *i* is view index *i*.

Never ``camera_positions.shape[0]`` -- a scene may legitimately carry more camera rows than view
folders, and the count feeds an RNG draw whose consumption depends on it.

One handle per scene read
-------------------------
:func:`open_scene` is a context manager whose lifetime should span a whole scene load: one open, one
central-directory parse, then N seeks. Opening per member would cost ~1.3 ms each on a 784-member
archive -- hundreds of ms per sample, on the code path this change exists to speed up. Nothing is
cached across calls: the handle must never outlive the call and never cross ``fork()``, and with a
shuffled sampler over thousands of scenes a cache would almost never hit anyway. Readers are
single-threaded per process.
"""
import contextlib
import io
import json
import os
import zipfile
from typing import Any, Dict, Iterator, List, Union

import numpy as np

VIEW_PREFIX = "view_"
ARCHIVE_SUFFIX = ".zip"


def _view_sort_key(name: str) -> int:
    """Order ``view_*`` names numerically, so ``view_10`` follows ``view_9`` and ``view_04`` works."""
    return int(name.split("_")[-1])


def _ordered_view_prefixes(names) -> List[str]:
    """Apply the module's one view-ordering rule to a collection of view-directory names."""
    return sorted({name for name in names if name.startswith(VIEW_PREFIX)}, key=_view_sort_key)


class SceneReader:
    """Common surface both layouts expose. ``reads`` records member names, for tests."""

    def __init__(self, label: str):
        self.label = label
        self.reads: List[str] = []

    def view_prefixes(self) -> List[str]:
        raise NotImplementedError

    def listdir(self, prefix: str) -> List[str]:
        """Sorted basenames directly under ``prefix``. Frame order comes from this sort."""
        raise NotImplementedError

    def exists(self, member: str) -> bool:
        raise NotImplementedError

    def read_bytes(self, member: str) -> bytes:
        raise NotImplementedError

    def load_npz(self, member: str) -> Any:
        raise NotImplementedError

    def read_json(self, member: str) -> Any:
        return json.loads(self.read_bytes(member))


class DirectoryScene(SceneReader):
    """A scene laid out as a directory -- the original format."""

    def __init__(self, scene_dir: Union[str, "os.PathLike[str]"]):
        super().__init__(os.fspath(scene_dir))
        self.scene_dir = os.fspath(scene_dir)

    def _path(self, member: str) -> str:
        return os.path.join(self.scene_dir, *member.split("/"))

    def view_prefixes(self) -> List[str]:
        return _ordered_view_prefixes(
            d for d in os.listdir(self.scene_dir) if os.path.isdir(self._path(d))
        )

    def listdir(self, prefix: str) -> List[str]:
        return sorted(os.listdir(self._path(prefix)))

    def exists(self, member: str) -> bool:
        return os.path.exists(self._path(member))

    def read_bytes(self, member: str) -> bytes:
        self.reads.append(member)
        with open(self._path(member), "rb") as f:
            return f.read()

    def load_npz(self, member: str) -> Any:
        # Straight from the path: no reason to route the loose layout through bytes, and doing so
        # would change peak memory for nothing.
        self.reads.append(member)
        return np.load(self._path(member))


class ZipScene(SceneReader):
    """A scene packed into one ZIP_STORED archive whose members mirror the directory tree."""

    def __init__(self, archive_path: Union[str, "os.PathLike[str]"], archive: zipfile.ZipFile):
        super().__init__(os.fspath(archive_path))
        self.archive = archive
        self._names = set(archive.namelist())  # duplicate member names are legal in ZIP; dedupe
        self._children: Dict[str, List[str]] = {}
        for name in self._names:
            head, sep, tail = name.partition("/")
            if sep and tail:
                self._children.setdefault(head, []).append(tail)
        self._reject_wrapped_archive()

    def _reject_wrapped_archive(self) -> None:
        """Refuse `zip -r <scene>.zip <scene>`, which nests everything under an extra directory.

        Reporting "zero views" instead would look exactly like a genuinely empty scene, and silently
        stripping the prefix would hide a half-packed pool that could then not be told from a correct
        one.
        """
        if any(head.startswith(VIEW_PREFIX) for head in self._children):
            return
        nested = sorted(
            head for head, tails in self._children.items()
            if any(tail.startswith(VIEW_PREFIX) for tail in tails)
        )
        if nested:
            raise ValueError(
                f"{self.label}: members are nested under {nested[0]!r} instead of being at the "
                f"archive root. Pack from inside the scene directory, not from its parent."
            )

    def view_prefixes(self) -> List[str]:
        return _ordered_view_prefixes(self._children)

    def listdir(self, prefix: str) -> List[str]:
        return sorted(name for name in self._children.get(prefix, []) if "/" not in name)

    def exists(self, member: str) -> bool:
        return member in self._names

    def read_bytes(self, member: str) -> bytes:
        self.reads.append(member)
        try:
            return self.archive.read(member)
        except KeyError:
            raise KeyError(f"{self.label}: archive has no member {member!r}") from None

    def load_npz(self, member: str) -> Any:
        # BytesIO, never the raw ZipExtFile. npz reading seeks backwards and ZipExtFile._seek rewinds
        # and re-reads to get there: measured 16.3 ms vs 5.6 ms on a 7.3 MB tracks_2d.npz. It *works*
        # through ZipExtFile, which is exactly why this looks like a pointless copy.
        return np.load(io.BytesIO(self.read_bytes(member)))


@contextlib.contextmanager
def open_scene(path: Union[str, "os.PathLike[str]"]) -> Iterator[SceneReader]:
    """Open a scene at ``path``, which is either a directory or ``<path>.zip``.

    Hold this open for a whole scene load -- see the module docstring on why the handle is scoped to
    one call and never cached.
    """
    path = os.fspath(path)
    archive_path = path if path.endswith(ARCHIVE_SUFFIX) else path + ARCHIVE_SUFFIX
    if not path.endswith(ARCHIVE_SUFFIX) and os.path.isdir(path):
        yield DirectoryScene(path)
        return
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"No scene at {path!r}: neither a directory nor {archive_path!r}")
    with zipfile.ZipFile(archive_path) as archive:
        yield ZipScene(archive_path, archive)


def loader_member_set(scene: SceneReader) -> List[str]:
    """Every member a full load of ``scene`` opens.

    This is the layout spec. The validator below checks exactly this set, and a test asserts it equals
    what a real load actually touched (``SceneReader.reads``) -- so the two cannot drift, which a
    comment asking someone to keep them in step cannot promise. Extend this when the loader starts
    reading something new, and both follow.

    ``segmentation_*.png``, ``data_ranges.json`` and ``events.json`` are deliberately absent: the
    loader substitutes ones for segmentation and never reads the other two, so a packer that drops
    them is correct.
    """
    members = ["tracks_3d.npz", "tracks_segmentation_ids.npz", "tracked_objects.json"]
    # Either camera file satisfies the loader; it prefers views.npz when both are present.
    members.append("views.npz" if scene.exists("views.npz") else "cameras.npz")
    for prefix in scene.view_prefixes():
        for name in scene.listdir(prefix):
            if name.startswith("rgba_") or name.startswith("depth_"):
                members.append(f"{prefix}/{name}")
        members += [f"{prefix}/tracks_2d.npz", f"{prefix}/metadata.json",
                    f"{prefix}/object_id_to_segmentation_id.json"]
    return members


def scene_view_count(path: Union[str, "os.PathLike[str]"]) -> int:
    """View count for one scene, for the construction-time scan. Opens and closes the archive."""
    with open_scene(path) as scene:
        return len(scene.view_prefixes())


def check_packed_root(data_root: str, expected_frames: int = 24, expected_views: int = 10) -> List[str]:
    """Sweep a packed pool and report every archive that does not match the documented layout.

    A 3.9M-file repack is not cheap to redo, so it is worth one pass over the real pool before
    training on it. Shares its layout expectations with the reader above and with the test packer in
    ``tests/kubric_scene_fixture.py``.

    Returns:
        One human-readable complaint per problem found; empty means the pool is well-formed.
    """
    problems: List[str] = []
    for entry in sorted(os.listdir(data_root)):
        if not entry.endswith(ARCHIVE_SUFFIX) or entry.startswith((".", "_")):
            continue
        archive_path = os.path.join(data_root, entry)
        try:
            with open_scene(archive_path) as scene:
                assert isinstance(scene, ZipScene)
                deflated = [i.filename for i in scene.archive.infolist()
                            if i.compress_type != zipfile.ZIP_STORED]
                if deflated:
                    problems.append(f"{entry}: {len(deflated)} member(s) not ZIP_STORED, "
                                    f"e.g. {deflated[0]}")
                prefixes = scene.view_prefixes()
                if len(prefixes) != expected_views:
                    problems.append(f"{entry}: {len(prefixes)} views, expected {expected_views}")
                for prefix in prefixes:
                    names = scene.listdir(prefix)
                    rgba = sum(n.startswith("rgba_") for n in names)
                    depth = sum(n.startswith("depth_") for n in names)
                    if rgba != expected_frames or depth != expected_frames:
                        problems.append(f"{entry}/{prefix}: {rgba} rgba, {depth} depth, "
                                        f"expected {expected_frames} of each")
                if not (scene.exists("cameras.npz") or scene.exists("views.npz")):
                    problems.append(f"{entry}: neither cameras.npz nor views.npz")
                # Every member a load would open, so a sweep cannot pass an archive that then dies
                # mid-training. Checking a hand-written subset is how tracked_objects.json and the
                # three per-view members went unvalidated.
                for member in loader_member_set(scene):
                    if not scene.exists(member):
                        problems.append(f"{entry}: missing {member}")
        except Exception as exc:  # a corrupt archive is a finding, not a crash
            problems.append(f"{entry}: {type(exc).__name__}: {exc}")
    return problems
