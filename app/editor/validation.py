"""Boundary validation for an ``EditorProject``.

``validate_project`` returns a list of human-readable error strings (empty means
valid) so callers can surface every problem at once rather than failing on the
first. It is pure — no I/O, no mutation.

Checks are split into two tiers by ``profile``:

* ``structural`` (default) — checks that guard against a broken or
  desynchronized render: overlaps, negative times, contiguity, atempo range,
  source-length overrun, unresolved media. These are correct for ANY client of
  a generic video editor.
* ``shorts`` — additionally enforces Shorts-specific editorial opinions this
  codebase's own n=6 exemplar research produced: audio coverage (no silent
  gaps), original-clip audio present, and the project's own min/max duration
  and clip-count bounds. A universal editor must not force one client's format
  opinions onto every agent, so these are opt-in.

Note there is no "must be vertical (9:16)" check at all — not even under
``shorts``: orientation is a client concern a caller can inspect via
``aspect_w``/``aspect_h`` on the project itself, not a structural or editorial
truth about video editing in general.
"""
from __future__ import annotations

from app.editor.model import (
    FIT_VALUES,
    AudioElement,
    ClipElement,
    EditorProject,
    OverlayElement,
)

VIDEO_TRACK_TYPE = "video"
TEXT_TRACK_TYPE = "text"
OVERLAY_TRACK_TYPE = "overlay"

# Float tolerance for timeline-boundary comparisons (contiguity, coverage).
# Without it, an arithmetic artifact as small as 1e-9s between a clip's end
# and the next clip's declared start (e.g. from summed float durations) reads
# as a real gap and produces a spurious error like "No audio covers
# [13.000000000000002, 13.0]s" that blocks an otherwise-clean export.
EPSILON = 1e-6

# ffmpeg's ``atempo`` filter (the only per-clip speed operator the render
# graph emits, see ``clip_audio_filter``) only accepts a factor in this range;
# outside it ffmpeg itself raises at render time. Catching it here turns that
# render-time failure into an edit-time one, the same shape as
# ``_check_media_resolves`` for a bogus media_id.
ATEMPO_MIN_SPEED = 0.5
ATEMPO_MAX_SPEED = 2.0


def _clip_elements(project: EditorProject) -> list[ClipElement]:
    return [
        element
        for track in project.tracks
        for element in track.elements
        if isinstance(element, ClipElement)
    ]


def _audio_elements(project: EditorProject) -> list[AudioElement]:
    return [
        element
        for track in project.tracks
        for element in track.elements
        if isinstance(element, AudioElement)
    ]


def _overlay_elements(project: EditorProject) -> list[OverlayElement]:
    return [
        element
        for track in project.tracks
        for element in track.elements
        if isinstance(element, OverlayElement)
    ]


def _video_tracks(project: EditorProject) -> list:
    return [track for track in project.tracks if track.type == VIDEO_TRACK_TYPE]


def _main_video_clips(project: EditorProject) -> list[ClipElement]:
    """Clip elements of the FIRST (V1/main) video track, in timeline order.

    Post-multitrack the renderer's base playlist (``mlt_graph.build_mlt_xml``'s
    ``playlist0``) is exactly this track's clips concatenated back-to-back, so
    contiguity and transitions are checked against it alone — NOT the old pool
    of every video track (V2+ tracks are explicitly allowed gaps)."""
    tracks = _video_tracks(project)
    if not tracks:
        return []
    clips = [el for el in tracks[0].elements if isinstance(el, ClipElement)]
    return sorted(clips, key=lambda clip: clip.start_time)


def _upper_video_tracks(project: EditorProject) -> list:
    """Every video track AFTER the first — the V2+ composited-over-V1 tracks."""
    return _video_tracks(project)[1:]


def _clip_source_end(clip: ClipElement) -> float:
    """Source out-point in seconds for ``clip``.

    Mirrors ``app.editor.ffmpeg_graph._clip_out_point`` (duplicated rather
    than imported: that name is module-private and validation must not reach
    into the render module's internals). Keep the two in sync by hand if
    trim/speed semantics ever change.
    """
    if clip.trim_end is not None:
        return clip.trim_end
    return clip.trim_start + clip.duration * clip.transform.speed


def _video_track_duration(project: EditorProject) -> float:
    """Latest end time across all video-track clips (0.0 when none)."""
    ends = [
        element.start_time + element.duration
        for track in project.tracks
        if track.type == VIDEO_TRACK_TYPE
        for element in track.elements
    ]
    return max(ends) if ends else 0.0


def _check_overlaps(project: EditorProject) -> list[str]:
    """Report any two elements that overlap in time on the same track.

    Text AND overlay tracks are exempt: both composite as a stack (a
    persistent header, a per-clip caption, and a PiP badge are all meant to be
    on screen together) — only clips (can't play two at once) and music beds
    are barred from overlapping.
    """
    errors: list[str] = []
    for track in project.tracks:
        if track.type in (TEXT_TRACK_TYPE, OVERLAY_TRACK_TYPE):
            continue
        ordered = sorted(track.elements, key=lambda el: el.start_time)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            earlier_end = earlier.start_time + earlier.duration
            if later.start_time < earlier_end:
                errors.append(
                    f"Overlapping elements on track {track.id!r}: "
                    f"{earlier.id!r} ends at {earlier_end} but {later.id!r} "
                    f"starts at {later.start_time}"
                )
    return errors


def _check_negative_times(project: EditorProject) -> list[str]:
    errors: list[str] = []
    for track in project.tracks:
        for element in track.elements:
            if element.start_time < 0:
                errors.append(
                    f"Element {element.id!r} has negative start_time "
                    f"{element.start_time}"
                )
            if element.duration is not None and element.duration < 0:
                errors.append(
                    f"Element {element.id!r} has negative duration "
                    f"{element.duration}"
                )
    return errors


def _audible_intervals(project: EditorProject) -> list[tuple[float, float]]:
    """Time spans that carry sound.

    A clip contributes its own span unless muted/silenced; a music bed
    contributes only as far as its source media actually reaches (a bed declared
    longer than its source runs dry — the exact failure that left video 1 silent
    after ~18s, so the tail beyond the source is NOT counted as covered)."""
    assets = {asset.id: asset for asset in project.assets}
    intervals: list[tuple[float, float]] = []
    for track in project.tracks:
        for element in track.elements:
            if isinstance(element, ClipElement) and track.type == VIDEO_TRACK_TYPE:
                if not element.muted and element.volume > 0:
                    intervals.append(
                        (element.start_time, element.start_time + element.duration)
                    )
            elif isinstance(element, AudioElement):
                asset = assets.get(element.media_id)
                source_len = asset.duration_sec if asset else None
                reach = (
                    element.duration
                    if source_len is None
                    else min(element.duration, source_len)
                )
                intervals.append((element.start_time, element.start_time + reach))
    return intervals


def _coverage_gaps(
    intervals: list[tuple[float, float]], end: float
) -> list[tuple[float, float]]:
    """Sub-spans of ``[0, end)`` not covered by any interval (merged).

    Boundary comparisons are epsilon-tolerant (see ``EPSILON``) so a
    sub-microsecond float artifact between two adjacent intervals — e.g. one
    clip's computed end landing at 13.000000000000002 while the next starts
    at 13.0 — is not reported as a silent gap.
    """
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, stop in sorted(intervals):
        if start > cursor + EPSILON:
            gaps.append((cursor, min(start, end)))
        cursor = max(cursor, stop)
        if cursor >= end - EPSILON:
            break
    if cursor < end - EPSILON:
        gaps.append((cursor, end))
    return gaps


def _check_audio_coverage(project: EditorProject) -> list[str]:
    """Report timeline spans with NO sound (silent gaps) and dry music beds.

    Shorts-editorial: a universal editor has no opinion on whether a client's
    timeline must stay fully covered by sound; this is opt-in under
    ``profile="shorts"``.
    """
    end = _video_track_duration(project)
    if end <= 0:
        return []
    return [
        f"No audio covers [{gap_start}, {gap_stop}]s — clips there are muted "
        f"and no music bed reaches it (silent gap)"
        for gap_start, gap_stop in _coverage_gaps(_audible_intervals(project), end)
    ]


def _check_original_audio(project: EditorProject) -> list[str]:
    """Report a timeline where NO clip keeps its own audio.

    Measured on 6 trending "Ranking" Shorts (2026-07-15): 6/6 keep the original
    clip audio audible — none is a dry music bed over muted sources. This is a
    Shorts-editorial opinion (opt-in under ``profile="shorts"``), not a
    structural truth about video editing.

    Integrated loudness (LUFS) and loudness range (LRA) are deliberately NOT
    checked here — they are properties of the decoded mix, unknowable from the
    project model. They belong to post-render acceptance (``ffmpeg loudnorm``).
    """
    clips = _clip_elements(project)
    if not clips:
        return []
    if any(not clip.muted and clip.volume > 0 for clip in clips):
        return []
    return [
        "No clip keeps its original audio (every clip is muted or at zero "
        "volume) — trending Shorts keep source sound audible; a music-only bed "
        "over muted clips is the refuted recipe"
    ]


def _check_video_contiguity(project: EditorProject) -> list[str]:
    """Report a video track that is not back-to-back from 0.

    ``_build_video_chain`` (the renderer) sorts video-track clips by
    ``start_time`` and concatenates them with no gap for one — every
    downstream consumer of TIMELINE time (``_video_timeline_end``, text
    ``enable=between(t,...)`` windows, music ``adelay`` offsets) assumes
    the clip declared at second N in the project IS the clip playing at
    second N of the render, which only holds when clips are contiguous. A
    declared gap does not render as a gap: it silently shifts every clip and
    overlay after it earlier by the gap's length, desynchronizing captions
    and music against the wrong footage. Since the renderer's contract is
    back-to-back concatenation and this module does not own the renderer,
    validation enforces that contract instead of changing it: each clip's
    ``start_time`` must equal the previous clip's end (within ``EPSILON``),
    and the first clip must start at 0.
    """
    errors: list[str] = []
    clips = _main_video_clips(project)
    if not clips:
        return errors

    first = clips[0]
    if abs(first.start_time) > EPSILON:
        errors.append(
            f"Video clip {first.id!r} starts at {first.start_time} instead "
            "of 0 — the renderer concatenates video-track clips back-to-back "
            "starting at 0"
        )

    for earlier, later in zip(clips, clips[1:], strict=False):
        earlier_end = earlier.start_time + earlier.duration
        if abs(later.start_time - earlier_end) > EPSILON:
            errors.append(
                f"Video clip {earlier.id!r} ends at {earlier_end} but "
                f"{later.id!r} starts at {later.start_time} — the video "
                "track must be contiguous (the renderer concatenates clips "
                "back-to-back and ignores any declared gap)"
            )
    return errors


def _check_clip_speed(project: EditorProject) -> list[str]:
    """Report a clip whose ``transform.speed`` is outside ffmpeg's atempo range.

    ``clip_audio_filter`` emits a single ``atempo={speed}`` stage per clip;
    ffmpeg's ``atempo`` filter only accepts a factor in
    ``[ATEMPO_MIN_SPEED, ATEMPO_MAX_SPEED]`` and raises outside it, which
    would otherwise surface as a render-time failure instead of an edit-time
    validation error."""
    return [
        f"Clip {clip.id!r} has transform.speed {clip.transform.speed}, "
        f"outside ffmpeg atempo's supported range "
        f"[{ATEMPO_MIN_SPEED}, {ATEMPO_MAX_SPEED}]"
        for clip in _clip_elements(project)
        if not (ATEMPO_MIN_SPEED <= clip.transform.speed <= ATEMPO_MAX_SPEED)
    ]


def _check_clip_source_length(project: EditorProject) -> list[str]:
    """Report a clip whose trim window reads past its source media's length.

    The analogous music-bed case is already caught in ``_audible_intervals``
    (a bed declared longer than its source runs dry, clamped with
    ``min(duration, source_len)``). A clip has the same source-length
    information available (``MediaAsset.duration_sec``, looked up the same
    way), but reading past it is worse than a dry patch: ffmpeg's
    ``trim``/``atrim`` simply stops at the source's last frame, so the
    rendered segment comes out SHORTER than the clip's declared ``duration``
    — desynchronizing every clip and overlay after it, the same failure
    shape as a declared gap. Only checked when ``duration_sec`` is known (it
    is ``None`` until the media is resolved), matching the AudioElement
    path — this is not invented data, just deferred until it exists.
    """
    assets = {asset.id: asset for asset in project.assets}
    errors: list[str] = []
    for clip in _clip_elements(project):
        asset = assets.get(clip.media_id)
        source_len = asset.duration_sec if asset else None
        if source_len is None:
            continue
        source_end = _clip_source_end(clip)
        if source_end > source_len + EPSILON:
            errors.append(
                f"Clip {clip.id!r} reads to {source_end}s of its source but "
                f"the source media is only {source_len}s long"
            )
    return errors


def _envelope_owners(project: EditorProject) -> list[ClipElement | AudioElement]:
    """Every clip + audio element, the two kinds that carry an audio
    envelope (``fade_in_sec``/``fade_out_sec``/``volume_keyframes``)."""
    return [*_clip_elements(project), *_audio_elements(project)]


def _check_audio_envelope(project: EditorProject) -> list[str]:
    """Report a fade/keyframe envelope that does not fit inside its own
    element's window: fades that together exceed the duration (the ramps
    would cross and the ``level`` keyframe track the mlt engine builds from
    them would go out of order — mltframework.org's animated properties
    require strictly increasing keyframe times), or a keyframe whose ``time``
    falls outside ``[0, duration]``.
    """
    errors: list[str] = []
    for element in _envelope_owners(project):
        if element.fade_in_sec < 0:
            errors.append(
                f"Element {element.id!r} has negative fade_in_sec {element.fade_in_sec}"
            )
        if element.fade_out_sec < 0:
            errors.append(
                f"Element {element.id!r} has negative fade_out_sec {element.fade_out_sec}"
            )
        if element.fade_in_sec + element.fade_out_sec > element.duration + EPSILON:
            errors.append(
                f"Element {element.id!r} fade_in_sec {element.fade_in_sec} + "
                f"fade_out_sec {element.fade_out_sec} exceeds its duration "
                f"{element.duration}"
            )
        for kf in element.volume_keyframes:
            if not -EPSILON <= kf.time <= element.duration + EPSILON:
                errors.append(
                    f"Element {element.id!r} has a volume keyframe at "
                    f"time {kf.time}, outside its window [0, {element.duration}]"
                )
            if kf.volume < 0:
                errors.append(
                    f"Element {element.id!r} has a volume keyframe with "
                    f"negative volume {kf.volume}"
                )
    return errors


def _check_clip_fit(project: EditorProject) -> list[str]:
    """Report a clip whose ``fit`` is not a recognized canvas mode."""
    return [
        f"Clip {clip.id!r} has fit {clip.fit!r}, must be one of {list(FIT_VALUES)}"
        for clip in _clip_elements(project)
        if clip.fit not in FIT_VALUES
    ]


def _check_media_resolves(project: EditorProject) -> list[str]:
    """Report any clip, music bed, OR overlay whose ``media_id`` has no
    matching asset.

    A bogus music media_id used to slip past this check (it only walked
    clips) and fail later, at render time, inside ``_resolve_media`` — turning
    an edit-time mistake into a render-time failure."""
    asset_ids = {asset.id for asset in project.assets}
    errors = [
        f"Clip {clip.id!r} references unknown media_id {clip.media_id!r}"
        for clip in _clip_elements(project)
        if clip.media_id not in asset_ids
    ]
    errors.extend(
        f"Audio {audio.id!r} references unknown media_id {audio.media_id!r}"
        for audio in _audio_elements(project)
        if audio.media_id not in asset_ids
    )
    errors.extend(
        f"Overlay {overlay.id!r} references unknown media_id {overlay.media_id!r}"
        for overlay in _overlay_elements(project)
        if overlay.media_id not in asset_ids
    )
    return errors


def _check_transitions(project: EditorProject) -> list[str]:
    """Report a transition whose overlap does not fit within both neighbors.

    ``TransitionSpec.duration`` (see ``ClipElement.transition_in``) consumes
    the last ``duration`` seconds of the PREVIOUS clip's own span and the
    first ``duration`` seconds of THIS clip's own span. Two independent
    bounds must hold:

    1. the duration cannot exceed either neighbor's own ``duration`` (there is
       not enough footage on that side to give up);
    2. a clip that is BOTH the target of one transition (consuming its own
       head) AND the predecessor of another (consuming its own tail) must
       have enough total duration for both halves at once — the two
       overlap zones must not collide inside that middle clip.
    """
    errors: list[str] = []
    clips = _main_video_clips(project)
    tail_consumed: dict[str, float] = {}
    for earlier, later in zip(clips, clips[1:], strict=False):
        transition = later.transition_in
        if transition is None:
            continue
        if transition.duration > earlier.duration:
            errors.append(
                f"Transition into clip {later.id!r} (duration {transition.duration}s) "
                f"exceeds predecessor {earlier.id!r}'s own duration {earlier.duration}s"
            )
        if transition.duration > later.duration:
            errors.append(
                f"Transition into clip {later.id!r} (duration {transition.duration}s) "
                f"exceeds its own duration {later.duration}s"
            )
        tail_consumed[earlier.id] = transition.duration

    for clip in clips:
        head_consumed = clip.transition_in.duration if clip.transition_in else 0.0
        consumed = head_consumed + tail_consumed.get(clip.id, 0.0)
        if consumed > clip.duration:
            errors.append(
                f"Clip {clip.id!r} (duration {clip.duration}s) cannot give up "
                f"{consumed}s to its neighboring transitions (head {head_consumed}s + "
                f"tail {tail_consumed.get(clip.id, 0.0)}s) without the two overlap "
                "zones colliding"
            )

    if clips and clips[0].transition_in is not None:
        errors.append(
            f"Clip {clips[0].id!r} is the first clip on its video track but "
            "has a transition_in set — there is no predecessor to dissolve from"
        )
    return errors


def _check_keyframes(project: EditorProject) -> list[str]:
    """Report a keyframe whose ``time`` falls outside its own clip's
    ``[0, duration]`` window (``TransformKeyframe.time`` is clip-relative,
    0 = the clip's first frame)."""
    errors: list[str] = []
    for clip in _clip_elements(project):
        for kf in clip.keyframes:
            if not (0.0 <= kf.time <= clip.duration + EPSILON):
                errors.append(
                    f"Keyframe at t={kf.time}s on clip {clip.id!r} is outside "
                    f"its own window [0, {clip.duration}]s"
                )
    return errors


def _check_upper_video_tracks(project: EditorProject) -> list[str]:
    """Report a V2+ (upper) video-track clip using a feature scoped to V1 only.

    Upper tracks carry the full clip toolset (trim/speed/color/volume/mute/
    envelope + placement via ``Transform`` scale/pos) EXCEPT the four features
    whose rect-vs-composite interplay is deferred (see the plan's locked
    semantics): ``transition_in`` and dissolves live only on the main track;
    ``keyframes``, ``fit="contain_blur"`` and ``crop_zoom`` are not expressible
    while the composite transition owns the clip's rect. ``opacity`` must be a
    real 0..1 mix. These are structural: the render cannot honor them here, so
    they block rather than silently drop."""
    errors: list[str] = []
    for track in _upper_video_tracks(project):
        for el in track.elements:
            if not isinstance(el, ClipElement):
                continue
            if el.transition_in is not None:
                errors.append(
                    f"Clip {el.id!r} on upper video track {track.id!r} has a "
                    "transition_in — transitions/dissolves are only allowed on "
                    "the first (V1) video track"
                )
            if el.keyframes:
                errors.append(
                    f"Clip {el.id!r} on upper video track {track.id!r} has "
                    "keyframes — animated transforms are not supported on upper "
                    "video tracks (the composite transition owns the rect)"
                )
            if el.fit == "contain_blur":
                errors.append(
                    f"Clip {el.id!r} on upper video track {track.id!r} uses "
                    "fit='contain_blur' — not supported on upper video tracks"
                )
            if el.transform.crop_zoom > 0:
                errors.append(
                    f"Clip {el.id!r} on upper video track {track.id!r} has "
                    "crop_zoom — not supported on upper video tracks"
                )
            if not (0.0 <= el.opacity <= 1.0):
                errors.append(
                    f"Clip {el.id!r} on upper video track {track.id!r} has "
                    f"opacity {el.opacity}, outside [0, 1]"
                )
    return errors


def upper_track_overhang_warnings(project: EditorProject) -> list[str]:
    """Advisory (NON-blocking): a V2+ clip whose span reaches past the first
    video track's end. V1 defines the film's length, so the tail beyond it is
    simply not rendered — not an error (the clip is otherwise legal), but the
    author should know the overhang is dropped. Deliberately NOT part of
    ``validate_project`` (which blocks export on any returned string); callers
    surface these separately (UI badge / render log)."""
    main = _main_video_clips(project)
    if not main:
        return []
    v1_end = max(clip.start_time + clip.duration for clip in main)
    warnings: list[str] = []
    for track in _upper_video_tracks(project):
        for el in track.elements:
            if not isinstance(el, ClipElement):
                continue
            el_end = el.start_time + el.duration
            if el_end > v1_end + EPSILON:
                warnings.append(
                    f"Clip {el.id!r} on upper video track {track.id!r} ends at "
                    f"{el_end} but the first (V1) video track ends at {v1_end} — "
                    "the overhang past V1 is not rendered (V1 defines length)"
                )
    return warnings


def validate_project(
    project: EditorProject,
    *,
    profile: str = "structural",
) -> list[str]:
    """Validate a project. Returns an empty list when valid, otherwise one
    string per violation.

    ``structural`` (default): the checks that guard against a broken or
    desynchronized render — overlaps, negative times, contiguity, atempo
    range, source-length overrun, unresolved media. Always correct for ANY
    client.

    ``shorts``: additionally enforces the Shorts-editorial rules this repo's
    n=6 exemplar research produced — audio coverage (no silent gaps),
    original-clip audio present, and the project's own min/max duration and
    clip-count bounds. Opt in explicitly; a universal editor must not force
    one client's format opinions onto every agent.
    """
    errors: list[str] = []
    errors.extend(_check_negative_times(project))
    errors.extend(_check_overlaps(project))
    errors.extend(_check_video_contiguity(project))
    errors.extend(_check_clip_speed(project))
    errors.extend(_check_clip_source_length(project))
    errors.extend(_check_media_resolves(project))
    errors.extend(_check_transitions(project))
    errors.extend(_check_keyframes(project))
    errors.extend(_check_audio_envelope(project))
    errors.extend(_check_clip_fit(project))
    errors.extend(_check_upper_video_tracks(project))

    if profile == "shorts":
        duration = _video_track_duration(project)
        if duration < project.min_duration_sec or (
            project.max_duration_sec is not None and duration > project.max_duration_sec
        ):
            errors.append(
                f"Video duration {duration}s is outside "
                f"[{project.min_duration_sec}, {project.max_duration_sec}]s"
            )
        clip_count = len(_clip_elements(project))
        if clip_count < project.min_clips:
            errors.append(
                f"Only {clip_count} clip(s); at least {project.min_clips} required"
            )
        if project.max_clips is not None and clip_count > project.max_clips:
            errors.append(
                f"{clip_count} clips exceed the maximum {project.max_clips}"
            )
        errors.extend(_check_audio_coverage(project))
        errors.extend(_check_original_audio(project))

    return errors
