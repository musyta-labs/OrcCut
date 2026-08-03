"""Pure, immutable mutation helpers for ``EditorProject``.

Every function takes a project and returns a NEW project with ``version + 1``,
never mutating the input (frozen dataclasses + tuple rebuilds). New element/
track/asset ids are ``uuid.uuid4().hex``. Unknown ids raise ``EditorError``.
"""
from __future__ import annotations

import uuid
from dataclasses import replace

from app.editor.errors import EditorError
from app.editor.model import (
    AudioElement,
    ClipElement,
    ColorAdjust,
    DEFAULT_TRANSITION_DURATION,
    Element,
    EditorProject,
    FIT_VALUES,
    MediaAsset,
    OverlayElement,
    TextStyle,
    Track,
    Transform,
    TransformKeyframe,
    TransitionSpec,
    TRANSITION_KINDS,
    VolumeKeyframe,
    metadata_from_dict,
)

DEFAULT_ASPECT = (1080, 1920)
DEFAULT_FPS = 30
DEFAULT_MUSIC_VOLUME = 0.15
DEFAULT_AUDIO_CLIP_VOLUME = 1.0
DEFAULT_TRACK_TYPE = "video"


def _new_id() -> str:
    return uuid.uuid4().hex


def _bump(project: EditorProject, **changes) -> EditorProject:
    """Return a copy of ``project`` with ``version + 1`` and ``changes`` applied."""
    return replace(project, version=project.version + 1, **changes)


def _find_element(project: EditorProject, element_id: str) -> tuple[Track, Element]:
    for track in project.tracks:
        for element in track.elements:
            if element.id == element_id:
                return track, element
    raise EditorError(f"Unknown element id: {element_id!r}")


def _replace_element(
    tracks: tuple[Track, ...], element_id: str, new_element: Element
) -> tuple[Track, ...]:
    return tuple(
        replace(
            track,
            elements=tuple(
                new_element if el.id == element_id else el for el in track.elements
            ),
        )
        for track in tracks
    )


def _track_end(track: Track) -> float:
    """Latest end time across a track's elements (0.0 when empty)."""
    if not track.elements:
        return 0.0
    return max(el.start_time + el.duration for el in track.elements)


def _tracks_with_appended(
    project: EditorProject, track_type: str, element: Element
) -> tuple[Track, ...]:
    """Append ``element`` to the first track of ``track_type``, creating that
    track when none exists yet."""
    for index, track in enumerate(project.tracks):
        if track.type == track_type:
            updated = replace(track, elements=track.elements + (element,))
            return project.tracks[:index] + (updated,) + project.tracks[index + 1 :]
    new_track = Track(id=_new_id(), type=track_type, elements=(element,))
    return project.tracks + (new_track,)


def _first_track_of_type(project: EditorProject, track_type: str) -> Track | None:
    for track in project.tracks:
        if track.type == track_type:
            return track
    return None


def _find_track(project: EditorProject, track_id: str) -> Track:
    """Return the track with ``track_id`` or raise ``EditorError``."""
    for track in project.tracks:
        if track.id == track_id:
            return track
    raise EditorError(f"Unknown track id: {track_id!r}")


def _tracks_with_appended_to(
    project: EditorProject, track_id: str, element: Element
) -> tuple[Track, ...]:
    """Append ``element`` to the EXISTING track with id ``track_id``.

    Unlike ``_tracks_with_appended`` (which targets the FIRST track of a type
    and creates one when absent), this targets one exact, already-created track
    — the primitive multitrack targeting needs, since a V2+ video track is
    minted explicitly (``add_video_track``) before anything lands on it. Raises
    ``EditorError`` when ``track_id`` is unknown."""
    for index, track in enumerate(project.tracks):
        if track.id == track_id:
            updated = replace(track, elements=track.elements + (element,))
            return project.tracks[:index] + (updated,) + project.tracks[index + 1 :]
    raise EditorError(f"Unknown track id: {track_id!r}")


def add_video_track(project: EditorProject) -> tuple[EditorProject, str]:
    """Add a new, empty video track (a V2+ upper track) and return
    ``(new_project, track_id)``. The FIRST video track (V1/main) is unaffected
    and stays first — z-order is track order, so this new track composites over
    V1 (and over any upper track already present). See
    ``app.editor.mlt_graph`` for how upper video tracks are emitted."""
    track = Track(id=_new_id(), type="video", elements=())
    # Insert right after the LAST video track, not at the tuple's end: lane
    # order in the timeline is storage order, so appending would render V2
    # below the text/audio lanes instead of directly under V1. Relative video
    # z-order (later video track = on top) is preserved either way.
    last_video = max(
        (i for i, t in enumerate(project.tracks) if t.type == "video"), default=-1
    )
    tracks = project.tracks[: last_video + 1] + (track,) + project.tracks[last_video + 1 :]
    return _bump(project, tracks=tracks), track.id


def add_audio_track(project: EditorProject) -> tuple[EditorProject, str]:
    """Add a new, empty audio track and return ``(new_project, track_id)``.

    Mirrors ``add_video_track`` exactly, for the audio track type: the new
    track is inserted right after the LAST audio track (not at the tuple's end)
    so lane order in the timeline stays storage order. When no audio track
    exists yet it lands at the front (``default=-1``), the same degenerate case
    ``add_video_track`` has. Overlap validation is per-track and audio is not
    exempt, so a second audio track is how a voiceover bed sidesteps a
    collision with the music bed on the first audio track."""
    track = Track(id=_new_id(), type="audio", elements=())
    last_audio = max(
        (i for i, t in enumerate(project.tracks) if t.type == "audio"), default=-1
    )
    tracks = project.tracks[: last_audio + 1] + (track,) + project.tracks[last_audio + 1 :]
    return _bump(project, tracks=tracks), track.id


def remove_track(project: EditorProject, track_id: str) -> EditorProject:
    """Remove a track and every element on it. Raises ``EditorError`` when
    ``track_id`` is unknown or names the FIRST (V1/main) video track — that
    track defines the film's length and contiguity and cannot be deleted (an
    empty timeline is a delete-the-clips operation, not a delete-the-track
    one)."""
    _find_track(project, track_id)  # raises when unknown
    first_video = _first_track_of_type(project, "video")
    if first_video is not None and first_video.id == track_id:
        raise EditorError(
            f"Track {track_id!r} is the first (V1/main) video track and cannot "
            "be removed"
        )
    tracks = tuple(track for track in project.tracks if track.id != track_id)
    return _bump(project, tracks=tracks)


def create_project(
    *,
    metadata: dict[str, str] | None = None,
    aspect: tuple[int, int] = DEFAULT_ASPECT,
    fps: int = DEFAULT_FPS,
    target_sec: float | None = None,
    min_clips: int = 1,
    max_clips: int | None = None,
    min_duration_sec: float = 0.0,
    max_duration_sec: float | None = None,
) -> EditorProject:
    """Create a fresh empty project (version 1). Tracks are created on demand
    by the add_* mutations."""
    aspect_w, aspect_h = aspect
    return EditorProject(
        id=_new_id(),
        metadata=metadata_from_dict(metadata),
        version=1,
        aspect_w=aspect_w,
        aspect_h=aspect_h,
        fps=fps,
        target_sec=target_sec,
        min_clips=min_clips,
        max_clips=max_clips,
        min_duration_sec=min_duration_sec,
        max_duration_sec=max_duration_sec,
        tracks=(),
        assets=(),
    )


def add_media(
    project: EditorProject,
    *,
    source: str,
    duration_sec: float | None = None,
) -> tuple[EditorProject, str]:
    """Register a media asset (a local path or URL). Returns (new_project, asset_id)."""
    asset = MediaAsset(
        id=_new_id(),
        source=source,
        duration_sec=duration_sec,
    )
    return _bump(project, assets=project.assets + (asset,)), asset.id


def set_media_duration(
    project: EditorProject, *, media_id: str, duration_sec: float
) -> EditorProject:
    """Backfill a probed ``duration_sec`` onto an already-registered asset — used
    once a URL asset has been downloaded and ffprobe can finally read it (a
    platform PAGE url has no probeable duration at ``add_media`` time). Returns a
    NEW project; raises ``EditorError`` if ``media_id`` is unknown."""
    found = False
    new_assets = []
    for asset in project.assets:
        if asset.id == media_id:
            new_assets.append(replace(asset, duration_sec=duration_sec))
            found = True
        else:
            new_assets.append(asset)
    if not found:
        raise EditorError(f"unknown media id: {media_id!r}")
    return _bump(project, assets=tuple(new_assets))


def add_clip(
    project: EditorProject,
    *,
    media_id: str,
    track_type: str = DEFAULT_TRACK_TYPE,
    start_time: float | None = None,
    duration: float,
    trim_start: float = 0.0,
    trim_end: float | None = None,
    track_id: str | None = None,
) -> tuple[EditorProject, str]:
    """Place a clip on a track. When ``start_time`` is None the clip is
    auto-appended after the last element on that track. Returns
    (new_project, element_id).

    ``track_id`` targets one EXACT, already-created track (a V2+ upper video
    track minted by ``add_video_track``); an unknown id raises ``EditorError``.
    When it is None, today's behavior stands: the first track of ``track_type``
    (created on demand)."""
    if track_id is not None:
        target = _find_track(project, track_id)  # raises when unknown
        if start_time is None:
            start_time = _track_end(target)
        element = ClipElement(
            id=_new_id(),
            media_id=media_id,
            start_time=start_time,
            duration=duration,
            trim_start=trim_start,
            trim_end=trim_end,
        )
        tracks = _tracks_with_appended_to(project, track_id, element)
        return _bump(project, tracks=tracks), element.id
    if start_time is None:
        existing = _first_track_of_type(project, track_type)
        start_time = _track_end(existing) if existing is not None else 0.0
    element = ClipElement(
        id=_new_id(),
        media_id=media_id,
        start_time=start_time,
        duration=duration,
        trim_start=trim_start,
        trim_end=trim_end,
    )
    tracks = _tracks_with_appended(project, track_type, element)
    return _bump(project, tracks=tracks), element.id


def trim_clip(
    project: EditorProject,
    element_id: str,
    *,
    trim_start: float | None = None,
    trim_end: float | None = None,
) -> EditorProject:
    """Adjust a clip's source in/out points. Passing neither is a no-op edit
    (still yields a new version)."""
    _, element = _find_element(project, element_id)
    if not isinstance(element, ClipElement):
        raise EditorError(f"Element {element_id!r} is not a clip")
    new_trim_start = element.trim_start if trim_start is None else trim_start
    new_trim_end = element.trim_end if trim_end is None else trim_end
    updated = replace(element, trim_start=new_trim_start, trim_end=new_trim_end)
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def resize_clip(
    project: EditorProject,
    element_id: str,
    *,
    start_time: float | None = None,
    duration: float | None = None,
    trim_start: float | None = None,
    trim_end: float | None = None,
) -> EditorProject:
    """Atomically edit a clip's timeline placement AND source window — the
    edge-drag gesture of a GUI editor, where shortening a clip must change
    ``duration``/``start_time`` together with ``trim_*`` in ONE journaled op
    (``trim_clip`` alone is a slip edit: it never touches placement). ``None``
    keeps the current value. Render-derived bounds (source length, overlaps)
    stay with ``validate_project``, per this module's convention."""
    _, element = _find_element(project, element_id)
    if not isinstance(element, ClipElement):
        raise EditorError(f"Element {element_id!r} is not a clip")
    if duration is not None and duration <= 0:
        raise EditorError(f"duration must be positive, got {duration}")
    updated = replace(
        element,
        start_time=element.start_time if start_time is None else start_time,
        duration=element.duration if duration is None else duration,
        trim_start=element.trim_start if trim_start is None else trim_start,
        trim_end=element.trim_end if trim_end is None else trim_end,
    )
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def split_clip(
    project: EditorProject, element_id: str, at_time: float
) -> EditorProject:
    """Split one clip into two adjacent clips at absolute timeline ``at_time``.
    The original element is replaced by two new elements in place."""
    track, element = _find_element(project, element_id)
    if not isinstance(element, ClipElement):
        raise EditorError(f"Element {element_id!r} is not a clip")
    clip_end = element.start_time + element.duration
    if not element.start_time < at_time < clip_end:
        raise EditorError(
            f"Split time {at_time} is outside clip "
            f"[{element.start_time}, {clip_end}]"
        )
    offset = at_time - element.start_time
    boundary = element.trim_start + offset
    first = replace(
        element,
        id=_new_id(),
        duration=offset,
        trim_end=boundary,
    )
    second = replace(
        element,
        id=_new_id(),
        start_time=at_time,
        duration=clip_end - at_time,
        trim_start=boundary,
        trim_end=element.trim_end,
    )
    new_elements: list[Element] = []
    for el in track.elements:
        if el.id == element_id:
            new_elements.append(first)
            new_elements.append(second)
        else:
            new_elements.append(el)
    tracks = tuple(
        replace(t, elements=tuple(new_elements)) if t.id == track.id else t
        for t in project.tracks
    )
    return _bump(project, tracks=tracks)


def cut_range(
    project: EditorProject,
    element_id: str,
    *,
    start_time: float,
    end_time: float,
) -> EditorProject:
    """Excise the middle range ``[start_time, end_time]`` from one clip in a
    single op — the "scissors" gesture. The clip becomes TWO adjacent clips:
    the material before ``start_time`` and the material after ``end_time``, with
    the trailing part shifted left to close the excised gap (so the two halves
    stay contiguous). Downstream elements are NOT rippled, mirroring
    ``split_clip``/``delete_element``.

    Source-window math follows ``split_clip``'s speed-agnostic convention: a
    timeline offset maps 1:1 onto a source (``trim_*``) offset. Both cut points
    must fall strictly inside the clip and ``start_time`` must precede
    ``end_time`` — otherwise ``EditorError`` (the caller's two handles can never
    cross, but the boundary is enforced here too)."""
    track, element = _find_element(project, element_id)
    if not isinstance(element, ClipElement):
        raise EditorError(f"Element {element_id!r} is not a clip")
    clip_end = element.start_time + element.duration
    if not element.start_time < start_time < end_time < clip_end:
        raise EditorError(
            f"Cut range [{start_time}, {end_time}] is not strictly inside clip "
            f"[{element.start_time}, {clip_end}]"
        )
    head_offset = start_time - element.start_time
    tail_offset = end_time - element.start_time
    first = replace(
        element,
        id=_new_id(),
        duration=head_offset,
        trim_end=element.trim_start + head_offset,
    )
    second = replace(
        element,
        id=_new_id(),
        start_time=start_time,  # shifted left to close the excised gap
        duration=clip_end - end_time,
        trim_start=element.trim_start + tail_offset,
        trim_end=element.trim_end,
    )
    new_elements: list[Element] = []
    for el in track.elements:
        if el.id == element_id:
            new_elements.append(first)
            new_elements.append(second)
        else:
            new_elements.append(el)
    tracks = tuple(
        replace(t, elements=tuple(new_elements)) if t.id == track.id else t
        for t in project.tracks
    )
    return _bump(project, tracks=tracks)


def move_clip(
    project: EditorProject,
    element_id: str,
    *,
    start_time: float,
    track_id: str | None = None,
) -> EditorProject:
    """Move an element to a new absolute start time, optionally onto a different
    track.

    ``track_id`` None (or the element's current track) keeps it in place and
    only changes ``start_time`` (today's behavior). A different, existing
    ``track_id`` moves the element there — stripped from its old track and
    APPENDED after that track's current elements (order-by-append, the same
    ordering ``add_clip`` gives). An unknown ``track_id`` raises
    ``EditorError``; the vertical-lane-drag gesture the GUI needs."""
    source_track, element = _find_element(project, element_id)
    updated = replace(element, start_time=start_time)
    if track_id is None or track_id == source_track.id:
        return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))
    _find_track(project, track_id)  # raises when unknown
    stripped = tuple(
        replace(track, elements=tuple(el for el in track.elements if el.id != element_id))
        for track in project.tracks
    )
    tracks = tuple(
        replace(track, elements=track.elements + (updated,)) if track.id == track_id else track
        for track in stripped
    )
    return _bump(project, tracks=tracks)


def delete_element(project: EditorProject, element_id: str) -> EditorProject:
    """Remove an element from whatever track holds it."""
    _find_element(project, element_id)  # raises if unknown
    tracks = tuple(
        replace(
            track,
            elements=tuple(el for el in track.elements if el.id != element_id),
        )
        for track in project.tracks
    )
    return _bump(project, tracks=tracks)


# Transform fields the ffmpeg graph reads directly: ``video_clip_filter``
# fill-scales + center-crops every clip to the full frame, a model with no
# notion of a sub-frame position or size to put ``scale``/``pos_x``/``pos_y``
# into. The MLT engine (default since phase 4) DOES apply all five fields —
# see ``app.editor.mlt_graph``'s affine builder — so ``set_transform`` no
# longer rejects scale/pos_x/pos_y (phase 3-4 guard, lifted in phase 4): a
# static (single-point, non-animated) scale/pos_x/pos_y is the degenerate
# case of a keyframed transform (``ClipElement.keyframes`` empty). Rendering
# such a project on the ffmpeg engine instead raises a clear ``EditorError``
# at render time — see ``app.editor.render._render_via_ffmpeg``'s guard —
# rather than silently dropping the fields.
SUPPORTED_TRANSFORM_FIELDS = ("scale", "pos_x", "pos_y", "speed", "crop_zoom")


def set_transform(
    project: EditorProject,
    element_id: str,
    *,
    scale: float | None = None,
    pos_x: float | None = None,
    pos_y: float | None = None,
    speed: float | None = None,
    crop_zoom: float | None = None,
    opacity: float | None = None,
) -> EditorProject:
    """Update a clip's transform. Only provided fields change.

    All five ``Transform`` fields are accepted (see ``SUPPORTED_TRANSFORM_FIELDS``):
    the MLT engine applies scale/pos_x/pos_y via an affine transform, static
    values being the single-keyframe degenerate case of ``ClipElement.keyframes``.
    The ffmpeg engine still cannot express them — that render path raises
    instead of silently ignoring them (``render._render_via_ffmpeg``).

    ``opacity`` (0..1) is a ClipElement field, not a ``Transform`` one — it is
    the composite mix for an UPPER (V2+) video-track clip (see
    ``ClipElement.opacity``); it is folded into this same tool so the canvas/
    inspector has one call for a clip's placement. ``None`` leaves it unchanged.
    """
    _, element = _find_element(project, element_id)
    if not isinstance(element, ClipElement):
        raise EditorError(f"Element {element_id!r} is not a clip")
    current = element.transform
    new_transform = Transform(
        scale=current.scale if scale is None else scale,
        pos_x=current.pos_x if pos_x is None else pos_x,
        pos_y=current.pos_y if pos_y is None else pos_y,
        speed=current.speed if speed is None else speed,
        crop_zoom=current.crop_zoom if crop_zoom is None else crop_zoom,
    )
    updated = replace(
        element,
        transform=new_transform,
        opacity=element.opacity if opacity is None else opacity,
    )
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def add_text(
    project: EditorProject,
    *,
    content: str,
    start_time: float,
    duration: float | None,
    style: TextStyle,
    role: str | None = None,
) -> tuple[EditorProject, str]:
    """Add a text overlay on the (auto-created) text track. ``role`` optionally
    marks the element's origin (e.g. ``"caption"`` for an auto-caption line).
    Returns (new_project, element_id)."""
    from app.editor.model import TextElement

    element = TextElement(
        id=_new_id(),
        content=content,
        start_time=start_time,
        duration=duration,
        style=style,
        role=role,
    )
    tracks = _tracks_with_appended(project, "text", element)
    return _bump(project, tracks=tracks), element.id


# Sentinel for update_* partial edits where ``None`` is itself a meaningful
# value (a text ``duration=None`` means "persistent header"): omitted = keep.
# Public: the tool layer forwards it across the JSON boundary (an omitted JSON
# key never reaches **kwargs, so defaults survive end-to-end).
UNSET: object = object()
_UNSET = UNSET


def update_text(
    project: EditorProject,
    element_id: str,
    *,
    content: str | None = None,
    start_time: float | None = None,
    duration: float | None | object = _UNSET,
    size: int | None = None,
    color: str | None = None,
    pos: str | None = None,
    pos_x: float | None | object = _UNSET,
    pos_y: float | None | object = _UNSET,
    font_id: str | None | object = _UNSET,
    font_path: str | None = None,
) -> EditorProject:
    """Partially edit a text element IN PLACE (id stable) — the GUI edit path.
    Without this, editing text meant delete+add: the id changed under any
    annotation pointing at it and undo grouping got two journal rows. ``None``
    is meaningful for ``duration`` (persistent), ``pos_x``/``pos_y`` (clear
    the normalized override back to the ``pos`` preset) and ``font_id`` (back
    to the configured font), so those use an omitted-keeps sentinel.

    ``font_id`` and ``font_path`` MOVE TOGETHER and are supplied by the caller
    as a pair: the id is the user's choice, the path is what the server resolved
    it to (``app.mcp.tools._font_resolver``). This layer is pure and must not
    look a font up itself — and it must never let the two drift apart, or the
    stored path would stop being derived from the stored id, which is the whole
    of R-33's guarantee."""
    from app.editor.model import TextElement

    _, element = _find_element(project, element_id)
    if not isinstance(element, TextElement):
        raise EditorError(f"Element {element_id!r} is not a text element")
    style = replace(
        element.style,
        size=element.style.size if size is None else size,
        color=element.style.color if color is None else color,
        pos=element.style.pos if pos is None else pos,
        pos_x=element.style.pos_x if pos_x is _UNSET else pos_x,
        pos_y=element.style.pos_y if pos_y is _UNSET else pos_y,
        font_id=element.style.font_id if font_id is _UNSET else font_id,
        font_path=element.style.font_path if font_path is None else font_path,
    )
    updated = replace(
        element,
        content=element.content if content is None else content,
        start_time=element.start_time if start_time is None else start_time,
        duration=element.duration if duration is _UNSET else duration,
        style=style,
    )
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def update_texts(
    project: EditorProject, *, updates: list[dict]
) -> EditorProject:
    """Atomically edit the ``content`` of MANY text elements in ONE version
    bump — the batch twin of ``update_text`` (content-only in v1). Each item is
    ``{"element_id": str, "content": str}``.

    ALL-OR-NOTHING: every ``element_id`` is resolved and type-checked BEFORE a
    single change is applied, so an unknown id or a non-text element raises
    ``EditorError`` with the input project left untouched — a caller (an agent
    translating a whole caption track) never ends up with a half-applied edit.
    Exactly one ``version + 1`` covers the whole batch."""
    from app.editor.model import TextElement

    # Resolve + validate the ENTIRE batch first; only then fold the changes in.
    replacements: list[tuple[str, TextElement]] = []
    for item in updates:
        if "element_id" not in item or "content" not in item:
            raise EditorError(
                "each update needs an 'element_id' and a 'content'; "
                f"got keys {sorted(item)}"
            )
        element_id = item["element_id"]
        _, element = _find_element(project, element_id)  # raises on unknown id
        if not isinstance(element, TextElement):
            raise EditorError(f"Element {element_id!r} is not a text element")
        replacements.append((element_id, replace(element, content=item["content"])))

    tracks = project.tracks
    for element_id, new_element in replacements:
        tracks = _replace_element(tracks, element_id, new_element)
    return _bump(project, tracks=tracks)


def update_overlay(
    project: EditorProject,
    element_id: str,
    *,
    start_time: float | None = None,
    duration: float | None = None,
    x: float | None = None,
    y: float | None = None,
    w: float | None = None,
    h: float | None = None,
    opacity: float | None = None,
) -> EditorProject:
    """Partially edit an overlay's placement/rect/opacity IN PLACE (id stable)
    — the canvas-drag path; see ``update_text`` for why delete+add was wrong.
    Rect/opacity bounds stay with the tool-layer validator, mirroring
    ``add_overlay``."""
    from app.editor.model import OverlayElement

    _, element = _find_element(project, element_id)
    if not isinstance(element, OverlayElement):
        raise EditorError(f"Element {element_id!r} is not an overlay")
    updated = replace(
        element,
        start_time=element.start_time if start_time is None else start_time,
        duration=element.duration if duration is None else duration,
        x=element.x if x is None else x,
        y=element.y if y is None else y,
        w=element.w if w is None else w,
        h=element.h if h is None else h,
        opacity=element.opacity if opacity is None else opacity,
    )
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def add_music(
    project: EditorProject,
    *,
    media_id: str,
    start_time: float = 0.0,
    duration: float,
    volume: float = DEFAULT_MUSIC_VOLUME,
) -> tuple[EditorProject, str]:
    """Add a music bed on the (auto-created) audio track. Returns
    (new_project, element_id)."""
    element = AudioElement(
        id=_new_id(),
        media_id=media_id,
        start_time=start_time,
        duration=duration,
        volume=volume,
    )
    tracks = _tracks_with_appended(project, "audio", element)
    return _bump(project, tracks=tracks), element.id


def add_audio_clip(
    project: EditorProject,
    *,
    media_id: str,
    track_id: str,
    start_time: float,
    duration: float,
    volume: float = DEFAULT_AUDIO_CLIP_VOLUME,
    gain_db: float = 0.0,
) -> tuple[EditorProject, str]:
    """Place an ``AudioElement`` on one EXACT, already-created audio track
    (``track_id``) — the targeted twin of ``add_music`` (which always lands on
    the FIRST audio track). This is what puts a synthesized voiceover clip on
    its own dedicated track (minted by ``add_audio_track``) so it never collides
    with the music bed on the first audio track. Unknown ``track_id`` raises
    ``EditorError``. Returns ``(new_project, element_id)``."""
    element = AudioElement(
        id=_new_id(),
        media_id=media_id,
        start_time=start_time,
        duration=duration,
        volume=volume,
        gain_db=gain_db,
    )
    tracks = _tracks_with_appended_to(project, track_id, element)
    return _bump(project, tracks=tracks), element.id


def set_clip_audio(
    project: EditorProject,
    element_id: str,
    *,
    muted: bool | None = None,
    volume: float | None = None,
    gain_db: float | None = None,
) -> EditorProject:
    """Update a clip's original-audio settings (mute / volume), OR a music
    bed's (``AudioElement``) volume / ``gain_db``. Only provided fields
    change.

    The two element kinds are NOT symmetric: a ``ClipElement`` has no
    ``gain_db`` field, so passing it for a clip raises ``EditorError``; an
    ``AudioElement`` has no ``muted`` field, so passing ``muted`` for a music
    bed raises ``EditorError`` too — it is never silently reinterpreted as
    ``volume=0``, because that would substitute a magic number for an
    explicit mute the model does not have.
    """
    _, element = _find_element(project, element_id)
    if isinstance(element, ClipElement):
        if gain_db is not None:
            raise EditorError(
                f"Element {element_id!r} is a clip; gain_db is music-bed (AudioElement) only"
            )
        new_muted = element.muted if muted is None else muted
        new_volume = element.volume if volume is None else volume
        updated = replace(element, muted=new_muted, volume=new_volume)
    elif isinstance(element, AudioElement):
        if muted is not None:
            raise EditorError(
                f"Element {element_id!r} is a music bed (AudioElement); it has no "
                "muted field — muted is clip-only and is never mapped to volume=0"
            )
        new_volume = element.volume if volume is None else volume
        new_gain_db = element.gain_db if gain_db is None else gain_db
        updated = replace(element, volume=new_volume, gain_db=new_gain_db)
    else:
        raise EditorError(f"Element {element_id!r} is not a clip or audio bed")
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


# --- CapCut toolset wave 1 (phase 4): transitions/keyframes/color/overlays --


def add_transition(
    project: EditorProject,
    element_id: str,
    *,
    kind: str = "dissolve",
    duration: float = DEFAULT_TRANSITION_DURATION,
) -> EditorProject:
    """Set a cross-dissolve from ``element_id``'s immediate predecessor on the
    (pooled, start_time-ordered) video track into it — see
    ``TransitionSpec``/``ClipElement.transition_in`` for the exact overlap
    semantics. MLT-engine only.

    Raises ``EditorError`` when ``element_id`` is not a clip, is not on a
    video-type track, is the FIRST clip on the pooled video track (no
    predecessor to dissolve from), ``kind`` is not a supported transition kind
    (see ``TRANSITION_KINDS``), or ``duration`` is not positive. The
    overlap-fits-in-both-neighbors bound (does this duration actually fit
    inside the previous clip's tail and this clip's head, accounting for any
    OTHER transition already claiming part of the same clip) is checked by
    ``app.editor.validation._check_transitions`` at validate time, not here —
    mirrors ``trim_clip``/``split_clip``: the mutation enforces structural
    invariants, ``validate_project`` enforces the render-derived bounds.
    """
    if kind not in TRANSITION_KINDS:
        raise EditorError(
            f"Unknown transition kind {kind!r}; supported: {list(TRANSITION_KINDS)}"
        )
    if duration <= 0:
        raise EditorError(f"transition duration must be positive, got {duration}")
    track, element = _find_element(project, element_id)
    if not isinstance(element, ClipElement):
        raise EditorError(f"Element {element_id!r} is not a clip")
    if track.type != "video":
        raise EditorError(f"Element {element_id!r} is not on a video track")
    video_clips = sorted(
        (
            el
            for t in project.tracks
            if t.type == "video"
            for el in t.elements
            if isinstance(el, ClipElement)
        ),
        key=lambda el: el.start_time,
    )
    index = next(i for i, el in enumerate(video_clips) if el.id == element_id)
    if index == 0:
        raise EditorError(
            f"Clip {element_id!r} is the first clip on its video track; "
            "no predecessor to dissolve from"
        )
    updated = replace(element, transition_in=TransitionSpec(kind=kind, duration=duration))
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def remove_transition(project: EditorProject, element_id: str) -> EditorProject:
    """Clear a clip's ``transition_in`` — the off switch ``add_transition``
    lacks (a GUI checkbox must be able to un-check). Clearing an already-empty
    transition is a valid no-op edit (still bumps the version)."""
    _, element = _find_element(project, element_id)
    if not isinstance(element, ClipElement):
        raise EditorError(f"Element {element_id!r} is not a clip")
    updated = replace(element, transition_in=None)
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def set_keyframes(
    project: EditorProject,
    element_id: str,
    *,
    keyframes: tuple[TransformKeyframe, ...],
) -> EditorProject:
    """Replace a clip's animated-transform keyframes, sorted by ``time``.
    Passing an empty tuple clears them (falls back to the clip's static
    ``transform`` scale/pos_x/pos_y — see ``ClipElement.keyframes``).
    MLT-engine only. Each keyframe's ``time`` must fall inside the clip's own
    ``[0, duration]`` window — checked by ``validate_project``, not here."""
    _, element = _find_element(project, element_id)
    if not isinstance(element, ClipElement):
        raise EditorError(f"Element {element_id!r} is not a clip")
    ordered = tuple(sorted(keyframes, key=lambda kf: kf.time))
    updated = replace(element, keyframes=ordered)
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def set_color(
    project: EditorProject,
    element_id: str,
    *,
    brightness: float | None = None,
    contrast: float | None = None,
    saturation: float | None = None,
    gamma: float | None = None,
) -> EditorProject:
    """Update a clip's color grade (rendered via MLT's ``avfilter.eq``
    bridge). Only provided fields change; starts from neutral defaults
    (``ColorAdjust()``) when the clip has no grade yet. MLT-engine only."""
    _, element = _find_element(project, element_id)
    if not isinstance(element, ClipElement):
        raise EditorError(f"Element {element_id!r} is not a clip")
    current = element.color or ColorAdjust()
    new_color = ColorAdjust(
        brightness=current.brightness if brightness is None else brightness,
        contrast=current.contrast if contrast is None else contrast,
        saturation=current.saturation if saturation is None else saturation,
        gamma=current.gamma if gamma is None else gamma,
    )
    updated = replace(element, color=new_color)
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def add_overlay(
    project: EditorProject,
    *,
    media_id: str,
    start_time: float | None = None,
    duration: float,
    x: float,
    y: float,
    w: float,
    h: float,
    opacity: float = 1.0,
) -> tuple[EditorProject, str]:
    """Place a PiP/sticker overlay (image or video) on the (auto-created)
    ``overlay`` track. When ``start_time`` is None the overlay auto-appends
    after the last element on that track (mirrors ``add_clip``). ``x``/``y``/
    ``w``/``h`` are normalized [0,1] frame fractions — bounds are validated at
    the MCP tool boundary (mirrors ``add_text``'s ``pos_x``/``pos_y``, not
    checked here). Returns (new_project, element_id). MLT-engine only."""
    if start_time is None:
        existing = _first_track_of_type(project, "overlay")
        start_time = _track_end(existing) if existing is not None else 0.0
    element = OverlayElement(
        id=_new_id(),
        media_id=media_id,
        start_time=start_time,
        duration=duration,
        x=x,
        y=y,
        w=w,
        h=h,
        opacity=opacity,
    )
    tracks = _tracks_with_appended(project, "overlay", element)
    return _bump(project, tracks=tracks), element.id
# --- audio envelope + canvas fit (phase 5 wave 2) ---------------------------

def set_audio_envelope(
    project: EditorProject,
    element_id: str,
    *,
    fade_in_sec: float | None = None,
    fade_out_sec: float | None = None,
    volume_keyframes: tuple[VolumeKeyframe, ...] | None = None,
) -> EditorProject:
    """Set a fade-in/fade-out and/or a volume-keyframe envelope on a clip's
    own audio OR a music bed. Only provided fields change. ``volume_keyframes``
    replaces the whole tuple (no partial merge) — pass the full envelope each
    time, mirroring how ``trim_clip`` treats each of its two fields.

    Structural correctness (fade lengths within the element's duration,
    keyframe times inside its window) is enforced by
    ``app.editor.validation``, not here — this mutation only writes the
    fields; ``editor_validate``/``editor_export`` catch a bad envelope before
    it reaches the render.
    """
    _, element = _find_element(project, element_id)
    if not isinstance(element, (ClipElement, AudioElement)):
        raise EditorError(f"Element {element_id!r} is not a clip or audio bed")
    new_fade_in = element.fade_in_sec if fade_in_sec is None else fade_in_sec
    new_fade_out = element.fade_out_sec if fade_out_sec is None else fade_out_sec
    new_keyframes = element.volume_keyframes if volume_keyframes is None else volume_keyframes
    updated = replace(
        element,
        fade_in_sec=new_fade_in,
        fade_out_sec=new_fade_out,
        volume_keyframes=new_keyframes,
    )
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def set_fit(project: EditorProject, element_id: str, *, fit: str) -> EditorProject:
    """Set a clip's canvas fill mode. ``"cover"`` (default) is today's
    fill-scale + center-crop; ``"contain_blur"`` letterboxes the source at
    full extent over a blurred, cover-scaled copy of itself (mlt engine only).
    """
    if fit not in FIT_VALUES:
        raise EditorError(f"fit must be one of {list(FIT_VALUES)}, got {fit!r}")
    _, element = _find_element(project, element_id)
    if not isinstance(element, ClipElement):
        raise EditorError(f"Element {element_id!r} is not a clip")
    updated = replace(element, fit=fit)
    return _bump(project, tracks=_replace_element(project.tracks, element_id, updated))


def set_cover(project: EditorProject, *, at_time: float) -> EditorProject:
    """Set the absolute timeline time exported as the project's cover PNG
    alongside the mp4 on the next ``export`` (see ``app.editor.render``)."""
    return _bump(project, cover_time=at_time)


def restore_snapshot(project: EditorProject, *, data: dict) -> EditorProject:
    """Undo/redo primitive: rebuild a project from a previously serialized
    snapshot and re-stamp it as the NEXT version of THIS project.

    The op journal is append-only with no inverse ops, but ``project_to_dict``
    round-trips losslessly, so 'undo' is simply persisting an older snapshot as
    a brand-new version. ``id`` is pinned to the live project (a snapshot may
    carry a foreign id) and ``version`` is strictly ``project.version + 1`` — it
    never rewinds, which the export-file max-version glob and the EXPORTED lock
    logic both rely on. Raises ``EditorError`` when ``data`` is not a valid
    snapshot (wrong shape / missing keys / a field whose VALUE contradicts its
    declared type — the dataclasses enforce nothing at runtime, so an unchecked
    snapshot is a way to put a crafted string into a render expression, R-04)."""
    from app.editor.serialization import project_from_dict
    from app.editor.typecheck import check_types

    try:
        restored = project_from_dict(data)
        check_types(restored)
    except (KeyError, TypeError, ValueError, EditorError) as exc:
        raise EditorError(f"invalid snapshot: {exc}") from exc
    return replace(restored, id=project.id, version=project.version + 1)
