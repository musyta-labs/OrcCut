"""Orchestrate clip event detection: decode → normalise → pick → render
keyframe strips, and hand back one frozen result.

This is the analysis package's only entry point, mirroring the shape of
``app.captions.build_captions.add_auto_captions`` — the ffmpeg work and the
math live in sibling modules, this file only wires them together and writes
the artefacts.

The result is DETERMINISTIC and LLM-free. The ``agent`` block of the contract
is left as all-nulls on purpose: only a live agent session with vision fills
it in, so a deterministic run can never be mistaken for a judged one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.analysis.audio import read_rms
from app.analysis.ffmpeg_io import probe_fps, run_ffmpeg_capture
from app.analysis.frames import motion_diffs, read_gray_frames
from app.analysis.peaks import (
    MAX_EVENTS,
    classify,
    combined_score,
    pick_events,
    zscore_over_baseline,
)
from app.common.logging import get_logger
from app.editor.render import probe_duration_sec
from app.media.guards import reject_pathological_media

# Artefact subdirectory under ``media_dir``, named the same way
# ``app.mcp.tools.VOICEOVER_SUBDIR`` names its own.
ANALYSIS_SUBDIR = "analysis"
# How the web layer serves that subdirectory. Kept next to the subdir name so
# the two cannot drift apart.
ANALYSIS_URL_PREFIX = "/ui/media/analysis"
# Contract version stamped into every result — clients read defensively
# against it (see the plan's "Annotation JSON contract").
#
# v2 (2026-07-20): every event gained a "score" field — the normalised value
# that actually caused its selection, and the key ``payoff_at`` now ranks by.
# "combined" is unchanged in meaning but is no longer the ranking key: events
# found by the new solo-channel rule (one channel above
# ``peaks.SOLO_Z_THRESHOLD`` while the other is quiet) carry a "combined" near
# zero, so consumers that sort or threshold on "combined" must move to
# "score". Consumers must also expect MORE events per clip and fewer clips
# typed "mood" than under v1.
ANALYSIS_VERSION = 2
# A keyframe strip shows the moment plus one second of context each side:
# enough for a viewer (human or vision model) to see what led in and what
# came out without watching the clip.
TRIPLET_OFFSET_SEC = 1.0
# Seconds format for filenames: zero-padded so names sort lexicographically
# and never start with a bare '.' (a shell/CLI footgun the spike hit).
_SECONDS_FORMAT = "05.2f"

logger = get_logger(__name__)


@dataclass(frozen=True)
class AnalysisResult:
    """Immutable outcome of one clip analysis. ``to_dict`` is the wire
    contract shared with the external annotation client."""

    duration_sec: float
    fps: float
    events: tuple[dict, ...]
    cuts: tuple[float, ...]
    clip_type: str
    payoff_at: float | None
    keyframes: tuple[Path, ...]

    def to_dict(self) -> dict:
        return {
            "version": ANALYSIS_VERSION,
            "duration_sec": self.duration_sec,
            "fps": self.fps,
            "events": [dict(event) for event in self.events],
            "cuts": list(self.cuts),
            "cut_count": len(self.cuts),
            "clip_type": self.clip_type,
            "payoff_at": self.payoff_at,
            # Basenames, NOT URLs (Gate 1 Step 10). Storage no longer hard-codes
            # a URL layout: the serving route builds the URL, and the route is
            # now ownership-checked, so the same stored record can be read by
            # every tenant legitimately holding these bytes while each gets
            # links only it may follow. Readers normalise old URL-shaped
            # entries via ``media_analysis.keyframe_name``, so no stored
            # analysis needed recomputing for this change.
            "keyframes": [path.name for path in self.keyframes],
            # Vision-only fields. Deterministic code MUST leave these null.
            "agent": {
                "needs_audio": None,
                "trim_hint": None,
                "ai_suspect": None,
                "why_it_works": None,
            },
        }


def payload_with_keyframe_urls(payload: dict) -> dict:
    """One stored payload with ``keyframes`` rendered as servable URLs.

    Storage holds basenames since Gate 1 Step 10, so a record does not
    hard-code the URL layout of whatever route happens to serve it. The WIRE
    contract is unchanged and still carries URLs — both the SPA and the
    annotation client read them — so the
    rendering happens here, at the boundary, for every caller that puts a
    payload on the wire.

    Entries stored in the old absolute-URL shape normalise to the same result,
    so no stored analysis had to be recomputed for the split. Returns a copy:
    the argument is often a live ORM attribute and must not be mutated.
    """
    from app.db.repositories.media_analysis import keyframe_names

    return {
        **payload,
        "keyframes": [
            f"{ANALYSIS_URL_PREFIX}/{name}" for name in keyframe_names(payload)
        ],
    }


def analyze_media_file(
    path: Path,
    *,
    media_dir: Path,
    analysis_id: str,
    max_events: int = MAX_EVENTS,
) -> AnalysisResult:
    """Analyse the media file at ``path`` and write its keyframe strips into
    ``media_dir/analysis/``.

    ``analysis_id`` is the ONLY caller-supplied part of any filename and is
    expected to be a uuid — no user input ever reaches the path. Raises
    ``EditorError`` when the file cannot be decoded (see ``read_gray_frames``)
    or when it fails ``reject_pathological_media``'s resolution/duration/
    stream-count bounds — checked AFTER the first ffprobe (``probe_fps``,
    timeout-guarded) but BEFORE the actual decode, so a small file with an
    absurd claimed shape never reaches ``read_gray_frames``.
    """
    path = Path(path)
    fps = probe_fps(path)
    reject_pathological_media(path)
    frames = read_gray_frames(path)
    motion = motion_diffs(frames)
    duration_sec = _duration_of(path, n_frames=frames.shape[0], fps=fps)

    audio = read_rms(path, fps=fps, n_frames=motion.size)
    motion_z = zscore_over_baseline(motion, fps=fps)
    audio_z = zscore_over_baseline(audio, fps=fps)
    combined = combined_score(motion_z, audio_z)
    # The RAW channels go through too, so ``pick_events`` can also measure each
    # moment against its own neighbourhood and catch a beat that a loud passage
    # elsewhere in the clip would otherwise hide — see its docstring's pool
    # measurement table.
    events = pick_events(
        combined,
        motion_z,
        audio_z,
        fps=fps,
        motion_raw=motion,
        audio_raw=audio,
        max_events=max_events,
    )
    clip_type, payoff_at, cuts = classify(events)

    keyframes = _write_keyframes(
        path,
        events=events,
        duration_sec=duration_sec,
        out_dir=Path(media_dir) / ANALYSIS_SUBDIR,
        analysis_id=analysis_id,
    )
    return AnalysisResult(
        duration_sec=round(duration_sec, 2),
        fps=round(fps, 3),
        events=tuple(events),
        cuts=tuple(cuts),
        clip_type=clip_type,
        payoff_at=payoff_at,
        keyframes=tuple(keyframes),
    )


def _duration_of(path: Path, *, n_frames: int, fps: float) -> float:
    """Prefer the container's own duration (``probe_duration_sec``, the
    established helper); fall back to the decoded frame count when the probe
    reports nothing."""
    probed = probe_duration_sec(str(path))
    if probed is not None and probed > 0:
        return float(probed)
    return n_frames / fps if fps > 0 else 0.0


def _write_keyframes(
    source: Path,
    *,
    events: list[dict],
    duration_sec: float,
    out_dir: Path,
    analysis_id: str,
) -> list[Path]:
    """An overview strip (first / middle / last) ALWAYS, then one strip per
    event (before / at / after).

    The overview is unconditional on purpose. It used to be a fallback behind
    ``if not events``, which meant it vanished the moment anything was
    detected — and after the v2 solo rule found events on most clips, a
    single-event clip returned exactly one strip covering t±1s. On a 10s clip
    with one event at 1.42s that is three near-identical frames of the first
    two seconds and no sight of the other 7.6s at all. The two strips answer
    different questions and neither replaces the other: the overview says what
    the clip IS, the triplet says what HAPPENS at a moment.

    Only strips that ACTUALLY landed on disk are returned (see
    ``_render_strip``). A payload must never advertise a keyframe URL for a
    file that does not exist: results are persisted now, so a phantom
    reference is permanent rather than re-rolled by the next analysis, and it
    would render as a broken image in the UI forever."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    overview_path = out_dir / f"{analysis_id}_overview.png"
    if _render_strip(source, _mood_times(duration_sec), overview_path) is not None:
        written.append(overview_path)

    for event in events:
        moment = float(event["t"])
        out_path = out_dir / f"{analysis_id}_t{moment:{_SECONDS_FORMAT}}.png"
        if _render_strip(source, _triplet_times(moment, duration_sec), out_path) is not None:
            written.append(out_path)
    return written


def _triplet_times(moment: float, duration_sec: float) -> list[float]:
    """``[t-1, t, t+1]`` clamped into the clip. An event at t<1 would otherwise
    seek to a negative timestamp (ffmpeg silently yields frame 0 or errors),
    and one near the end would seek past EOF."""
    last = max(duration_sec - _last_frame_margin(duration_sec), 0.0)
    return [
        min(max(moment + offset, 0.0), last)
        for offset in (-TRIPLET_OFFSET_SEC, 0.0, TRIPLET_OFFSET_SEC)
    ]


def _mood_times(duration_sec: float) -> list[float]:
    last = max(duration_sec - _last_frame_margin(duration_sec), 0.0)
    return [0.0, last / 2, last]


def _last_frame_margin(duration_sec: float) -> float:
    """Back off a hair from the very end: seeking exactly to ``duration`` lands
    past the last frame and decodes nothing."""
    return min(0.1, duration_sec / 10)


def _render_strip(source: Path, times: list[float], out_path: Path) -> Path | None:
    """hstack ``times`` worth of stills from ``source`` into one PNG. Each
    ``-ss`` precedes its own ``-i`` so ffmpeg seeks the input (fast) rather
    than decoding from zero for every still.

    Returns ``None`` when ffmpeg produced no file. A zero exit code is NOT
    proof the strip exists: when a seek lands past the last decodable frame
    that input yields no frames, ``hstack`` never produces one, and ffmpeg
    exits 0 having written nothing. This is reachable in ordinary use because
    the container duration can overstate the last real frame — measured live
    2026-07-20 on a clip whose container reported 7.079s while its last
    decodable frame was at 6.267s, so the overview's final still (6.98s) came
    back empty and the analysis advertised a PNG that did not exist.

    Reported rather than raised: a missing strip is cosmetic — the events,
    cuts and payoff timestamp are the valuable part and are already computed —
    so it must not fail the whole analysis. It must simply not be claimed."""
    argv = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for moment in times:
        argv += ["-ss", f"{moment:.3f}", "-i", str(source)]
    inputs = "".join(f"[{index}:v]" for index in range(len(times)))
    argv += [
        "-filter_complex", f"{inputs}hstack=inputs={len(times)}[out]",
        "-map", "[out]",
        "-frames:v", "1",
        str(out_path),
    ]
    run_ffmpeg_capture(argv, what=f"rendering keyframe strip {out_path.name!r}")
    if not out_path.is_file():
        logger.warning(
            "keyframe strip %s was not produced (ffmpeg exited 0 but wrote no "
            "file — a seek at %s likely landed past the last decodable frame); "
            "omitting it from the analysis payload rather than advertising a "
            "missing image",
            out_path.name, [round(t, 3) for t in times],
        )
        return None
    return out_path
