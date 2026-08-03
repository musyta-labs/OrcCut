"""R-20 — the free-space floor, and the ``ENOSPC`` that gets through anyway.

Every ceiling this service had before bounded a TENANT: operations per day,
concurrent renders, bytes on a personal shelf, bytes of download cache. None of
them bounds the HOST. N tenants each comfortably inside their own limits still
add up, and a burst inside the six-hour window between retention sweeps fills
the volume — after which every write fails HALFWAY THROUGH. That is the failure
this module exists to convert into a refusal: a render dying at minute fourteen
of a fifteen-minute ffmpeg run costs the compute AND leaves the operator reading
a stack trace, where an honest "no room, try later" costs nothing and says what
is wrong.

WHAT IS MEASURED, AND WHY IT IS A PATH RATHER THAN A STORE. The check reads the
filesystem holding the path a write is ABOUT TO TOUCH. That is always a local
one: ffmpeg/melt render to ``media_dir/exports/…``, yt-dlp downloads into
``media_dir/clips/…``, an upload streams into ``media_dir/uploads/…``. On the
``ARTIFACT_BACKEND=s3`` deployment those writes still happen — the object store
is fed from that staging area afterwards — so the guard stays exactly as
meaningful there, and a guard that switched itself off for S3 would leave the
staging volume as unprotected as it was before R-20.

What is NOT measured, ever, is the BUCKET. "Free space" is not a question an
object store answers, so this module never asks it: there is no store argument,
no backend branch, and no S3 call on the admission path. The honest reading of
a local volume is offered; the dishonest pretence of a remote one is not.

WHY A FLOOR AND NOT A PREDICTION. The guard does not try to estimate how many
bytes the coming render or download will need. It could not: an encode's output
size is not knowable from its inputs, and a wrong estimate would refuse work
that would have fit. A floor answers the question that is actually decidable —
"is there enough room left that this is worth starting at all" — and leaves the
size of the headroom to the operator, who knows their volume.

THE RACE IS REAL AND IS NOT PRETENDED AWAY. Free space is read at one instant
and written at another; between them another tenant's render can take the
remainder. ``ENOSPC`` therefore stays possible no matter how high the floor,
which is why this module also owns ``is_out_of_space``: the writers that can
hit it (``app.storage.local``, ``app.editor.render``) catch it, remove whatever
partial artifact they had begun, and re-raise it as the same sentence the floor
would have produced. A caller must never have to tell "the disk filled up" from
"the encoder crashed" by reading an errno.
"""
from __future__ import annotations

import errno
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings
from app.editor.errors import EditorError

__all__ = [
    "InsufficientDiskSpaceError",
    "free_bytes",
    "guard_disk_space",
    "is_out_of_space",
    "no_space_becomes_a_sentence",
    "out_of_space_error",
]

# ENOSPC is the volume itself being full; EDQUOT is a filesystem quota being
# spent (NFS, XFS project quotas, a container's own volume limit). They are the
# same event from every caller's point of view — the write cannot complete and
# retrying it immediately will not help — so they are treated as one.
_OUT_OF_SPACE_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})


class InsufficientDiskSpaceError(EditorError):
    """The server has no room to write what was asked for.

    Subclasses ``EditorError`` so it rides the existing error plumbing, and is a
    DISTINCT type from every quota error because it means the opposite thing
    about WHOSE fault it is: a quota says "you have used your share", this says
    "the server is out of room, and nothing you delete will change that". The
    HTTP boundary maps it to 507 by type, never by sniffing the message.
    """


def free_bytes(path: Path | str) -> int | None:
    """Free bytes on the filesystem holding ``path``, or ``None`` when the
    volume cannot be read.

    Walks UP to the nearest existing ancestor, because the guard runs BEFORE the
    write and the directory it is asked about may not exist yet (``exports/`` on
    a fresh volume). An ancestor is on the same filesystem in every layout this
    app creates, so it answers the same question.

    ``None`` — not zero — when ``disk_usage`` raises. An unreadable volume is
    not a full one, and answering "0 free" would turn a diagnostic failure into
    a total outage.
    """
    candidate = Path(path)
    for target in (candidate, *candidate.parents):
        if not target.exists():
            continue
        try:
            return shutil.disk_usage(target).free
        except OSError:
            return None
    return None


def _human_bytes(value: int) -> str:
    """Bytes in the unit an operator would use for them — the same rounding
    ``app.web.free_tier`` uses for its own refusals, so two limits reported in
    the same session never look like they came from different products."""
    if value >= 1024 ** 3:
        return f"{value / 1024 ** 3:.1f} GiB"
    if value >= 1024 ** 2:
        return f"{value / 1024 ** 2:.1f} MiB"
    return f"{value} bytes"


def guard_disk_space(path: Path | str, *, purpose: str) -> None:
    """THE free-space checkpoint. Raise ``InsufficientDiskSpaceError`` when the
    volume holding ``path`` has less than ``Settings.min_free_disk_bytes`` free.

    A no-op when no floor is configured (the self-host default, so an existing
    installation upgrades into unchanged behaviour) and a no-op when the volume
    cannot be read, so every surface may call it unconditionally — and should:
    a surface that skips it is a surface where writes still die mid-stream.

    ``purpose`` names the work being refused ("render", "upload", …) and appears
    in the message. It is a word for a human, not a discriminator — nothing
    branches on it.
    """
    floor = get_settings().min_free_disk_bytes
    if floor < 1:
        return
    available = free_bytes(path)
    if available is None or available >= floor:
        return
    raise InsufficientDiskSpaceError(
        f"the server is low on disk space and cannot start this {purpose}: "
        f"{_human_bytes(available)} free, {_human_bytes(floor)} required. This "
        "is the server's own storage, not your quota — nothing you delete will "
        "free it. Finished exports and cached clips are cleared automatically, "
        "so this usually resolves on its own; if it does not, the operator has "
        "to add space."
    )


def is_out_of_space(exc: BaseException) -> bool:
    """Whether ``exc`` is the disk (or a filesystem quota) running out.

    The ONE place that question is answered, so a caller cannot check
    ``ENOSPC`` alone on one path and both errnos on another. Anything that is
    not an ``OSError`` is not this — a permission problem dressed as an
    out-of-space message would send an operator hunting for disk they already
    have.
    """
    return isinstance(exc, OSError) and exc.errno in _OUT_OF_SPACE_ERRNOS


def out_of_space_error(purpose: str, exc: OSError) -> InsufficientDiskSpaceError:
    """The sentence to re-raise an ``ENOSPC`` as, for the writers that lose the
    race ``guard_disk_space`` cannot close.

    Deliberately worded like the floor's own refusal: from the caller's side the
    two are the same event, distinguishable only by whether the server noticed
    before or during the write, and there is no action that differs between
    them.
    """
    detail = exc.strerror or str(exc)
    return InsufficientDiskSpaceError(
        f"the server ran out of disk space during this {purpose} and it was "
        f"rolled back ({detail}). This is the server's own storage, not your "
        "quota — nothing you delete will free it. Retrying shortly is safe; "
        "the partially written file has been removed."
    )


@contextmanager
def no_space_becomes_a_sentence(
    purpose: str, *, cleanup: Path | None = None
) -> Iterator[None]:
    """Re-raise a write that died on a full volume as ``out_of_space_error``,
    after removing ``cleanup`` if a fragment was left at a known path.

    The half of R-20 that handles the race rather than the floor, in ONE place
    so a writer cannot get "which errnos count" or "clean up first, then raise"
    subtly different from its neighbour. Everything that is not an out-of-space
    error propagates untouched — an operator sent hunting for disk they already
    have is worse off than one reading the real errno.

    ``cleanup`` is optional because the render paths already unlink their own
    ``.tmp.mp4``/``.tmp.png`` in a ``finally``: by the time this sees the error
    there is nothing left to remove, and naming the file again here would be a
    second, drifting copy of that knowledge.
    """
    try:
        yield
    except OSError as exc:
        if not is_out_of_space(exc):
            raise
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)
        raise out_of_space_error(purpose, exc) from exc
