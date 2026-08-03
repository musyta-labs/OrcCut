"""Pure math over the motion/audio signals: normalisation, peak picking,
classification. Zero subprocess, zero I/O — every function here takes arrays
and returns new arrays or new dicts.

THRESHOLDS ARE MEASURED, NOT GUESSED. Every constant below was validated on
live clips (2026-07-20 spike, four cats clips): real actions score z 2.5-5.5
while a hard editorial cut scores 14+, which is exactly why one threshold can
separate "something happened" from "the editor spliced here". Do not tune
these without re-measuring on real footage.

RE-MEASURED 2026-07-20 on the full 23-clip pool. The original spike's four
clips all happened to peak in BOTH channels on the same frame, which hid the
fact that ``combined`` alone is a coincidence detector: across the real pool
14 of 23 clips had zero actions, several with motion_z above 5 that the
geometric mean crushed below 2.5 (worst case motion_z 5.57 -> combined 1.02).
Hence ``SOLO_Z_THRESHOLD``: one channel strong on its own also makes an event.
"""
from __future__ import annotations

import numpy as np

# A sample must stand this many standard deviations above its local baseline
# to count as an event at all.
Z_THRESHOLD = 2.5
# A SINGLE channel this far above its baseline is an event on its own, even
# when the other channel is quiet at that instant — the thing was seen OR
# heard clearly. MEASURED over the 23-clip pool: 5.0 is the lowest value that
# still resolves the reference "rocket" clip (motion_z 5.57 at 11.93s, audio
# peaking 6s away) to exactly ONE event; 4.5 blows the same clip up to 3, and
# 6.0 misses its 5.57 peak entirely. It also leaves real margin under that
# 5.57 (5.5 would clear it by 0.07 — inside decode noise) and sits at the top
# of the measured action band 2.5-5.5, safely under CUT_Z_THRESHOLD.
SOLO_Z_THRESHOLD = 5.0
# Above this, the "motion" is a whole-frame content swap — a cut, not an
# action. Actions measured 2.5-5.5; cuts measured 14+.
CUT_Z_THRESHOLD = 8.0
# Non-maximum suppression radius: two peaks closer than this describe the same
# moment, so only the stronger survives.
MIN_GAP_SEC = 0.7
# Rolling window the z-score is measured against. Long enough to average out
# one action, short enough to track a changing shot.
BASELINE_SEC = 2.0
# Floor under the LOCAL spread estimate, as a fraction of the clip's own global
# spread. Without it a dead-quiet passage has a near-zero local denominator and
# every ripple in it scores infinitely high. MEASURED over the 23-clip pool:
# anything in 0.25-0.80 gives an identical 39 events pool-wide; 1.00 collapses
# the estimator back onto the global spread and loses the moment this rule
# exists for. 0.5 is the middle of that plateau.
LOCAL_SPREAD_FLOOR_FRAC = 0.5
# Cap on reported events: a Short's worth of "moments" is a handful; more than
# this is noise, not signal.
MAX_EVENTS = 4
# Below this spread the audio channel carries no information (silence, or a
# constant tone) and the combined score falls back to motion only.
FLAT_SIGNAL_STD = 1e-6

EVENT_KIND_ACTION = "action"
EVENT_KIND_CUT = "cut"
CLIP_TYPE_EVENT = "event"
CLIP_TYPE_MOOD = "mood"


def zscore_over_baseline(
    signal: np.ndarray, *, fps: float, baseline_sec: float = BASELINE_SEC
) -> np.ndarray:
    """Standard scores of ``signal`` against its own rolling local mean.

    A global mean would be dominated by whichever shot is longest; measuring
    each sample against the ~2s around it is what makes one threshold work
    across clips of different pace. Returns a new array of the same length.

    KNOWN LIMITATION — measured 2026-07-20, deliberately NOT fixed. The
    baseline rolls, but ``spread`` is a single ``std`` over the WHOLE clip, so
    a lone spike inflates its own denominator and short clips have a hard
    z ceiling no content can beat. Measured with one ideal impulse at 30fps::

        0.5s -> 3.94   1.0s -> 5.48   2.0s ->  7.68   2.5s ->  8.59
        5.0s -> 12.14  7.0s -> 14.37  13.0s -> 19.58  27.0s -> 28.22

    So ``SOLO_Z_THRESHOLD`` (5.0) is unreachable below ~1s and
    ``CUT_Z_THRESHOLD`` (8.0) below ~2.2s, whatever the footage.

    This does NOT bite on real input and the fix (rolling or MAD-based spread)
    was therefore not applied. Measured over the live 10-clip annotated pool:
    analysable durations run 6.27-18.54s, giving ceilings of 13.60-33.19 —
    0 of 10 clips sit below EITHER bar, the tightest still clearing the 8.0
    cut bar by 1.7x, while observed motion_z runs 3.49-16.85 (i.e. real clips
    do reach past the cut bar, so the ceiling is not what is limiting them).
    The production pool floor is 5s, whose 12.14 ceiling clears 8.0 by 1.5x;
    a clip short enough to bite (<2.2s) is one this pipeline never produces.
    Revisit only if sub-2s media becomes a real input.
    """
    if signal.size == 0:
        return np.zeros(0, dtype=np.float64)
    window = max(int(round(baseline_sec * fps)), 1) if fps > 0 else 1
    baseline = _rolling_mean(signal, window)
    residual = signal - baseline
    spread = float(np.std(residual))
    if spread < FLAT_SIGNAL_STD:
        return np.zeros(signal.size, dtype=np.float64)
    return residual / spread


def zscore_over_local_spread(
    signal: np.ndarray,
    *,
    fps: float,
    baseline_sec: float = BASELINE_SEC,
    floor_frac: float = LOCAL_SPREAD_FLOOR_FRAC,
) -> np.ndarray:
    """Standard scores of ``signal`` against a spread measured LOCALLY, the
    same ~2s window the baseline already rolls over. Returns a new array.

    The twin of ``zscore_over_baseline``, differing only in the denominator,
    and it exists to answer one measured complaint. On the reference rocket
    clip (2026-07-20, operator-reported) the cat boards at 3.4-5.5s and the
    hatch slams at 5.65s, but the rocket's takeoff and the woman's shriek fill
    the clip's second half. Measured on that clip's audio residual::

        std over 0-8s  (calm, pre-ignition) = 1074.6  -> audio_z@5.65 = 6.54
        std over 8-15s (takeoff + shriek)   = 2176.1     (2.02x the calm half)
        std over the whole clip             = 1681.1  -> audio_z@5.65 = 4.18

    One global std is thus a compromise between two halves that is wrong for
    both: it puts the boarding beat at 4.18, under ``SOLO_Z_THRESHOLD``, so the
    detector could not see a moment the operator names unprompted. Against the
    local spread the same sample measures 5.30 and clears the bar.

    THIS IS AN ADDITIONAL RULE, NEVER A REPLACEMENT — measured, and the
    measurement is the whole argument. Swapping the global estimator for this
    one wholesale was tried on the 23-clip pool and REJECTED: it also divides
    the takeoff (local spread 1945) and the shriek (2789) by their own inflated
    neighbourhoods, so the reference clip drops from 2 events to 1 and keeps
    only the boarding beat, losing both real payoffs; pool-wide it falls 38 ->
    21 events with mood clips rising 6 -> 8 of 23. A locally normalised score
    is a RELATIVE detector: it surfaces quiet-region moments by exactly the
    construction that suppresses loud-region ones, and a clip's punchline
    usually lives in the loud region. Used as a union (see ``pick_events``) it
    adds the first without costing the second.
    """
    if signal.size == 0:
        return np.zeros(0, dtype=np.float64)
    window = max(int(round(baseline_sec * fps)), 1) if fps > 0 else 1
    residual = np.asarray(signal, dtype=np.float64) - _rolling_mean(signal, window)
    global_spread = float(np.std(residual))
    if global_spread < FLAT_SIGNAL_STD:
        return np.zeros(signal.size, dtype=np.float64)
    local = np.sqrt(np.maximum(_rolling_mean(residual**2, window), 0.0))
    return residual / np.maximum(local, floor_frac * global_spread)


def _rolling_mean(signal: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, edge-padded so the result keeps the input's
    length and the first/last samples are not dragged toward zero."""
    if window <= 1:
        return np.array(signal, dtype=np.float64)
    half = window // 2
    padded = np.pad(signal.astype(np.float64), (half, half), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")[: signal.size]


def combined_score(motion_z: np.ndarray, audio_z: np.ndarray) -> np.ndarray:
    """Geometric mean of the two positive z-signals.

    Geometric (not arithmetic) so a moment must be loud AND busy to score
    high — a loud static shot or a silent pan each stay low. Falls back to
    motion alone when the audio channel is flat (no audio stream, or silence).

    KNOWN LIMITATION — measured 2026-07-20, deliberately NOT fixed. That
    fallback changes what the ``z_threshold`` bar is applied TO: with audio,
    a frame needs ``sqrt(m * a) >= 2.5``; without it, ``m >= 2.5`` directly.
    The same footage muxed with a quiet ambient track and as video-only can
    therefore classify differently — the video-only cut is the EASIER one to
    detect on, which is the counter-intuitive direction.

    Note the trigger is sharper than it looks: ``audio_z`` comes out of
    ``zscore_over_baseline``, so its std is exactly 1.0 whenever the channel
    carried any information at all and exactly 0.0 when it did not. The
    ``FLAT_SIGNAL_STD`` test is thus a clean binary "was there an audio signal
    at all", never a judgement call on a quiet track.

    Not fixed because it does not occur: measured over the live 10-clip
    annotated pool, 10 of 10 clips carry an aac audio stream and 0 of 10 take
    this fallback (every one measured audio std exactly 1.0). Clips reach this
    engine from TikTok/YouTube sources that always carry audio. Revisit if a
    silent-source path (generated b-roll, stripped audio) ever feeds it.
    """
    positive_motion = np.clip(motion_z, 0.0, None)
    if audio_z.size != motion_z.size or float(np.std(audio_z)) < FLAT_SIGNAL_STD:
        return positive_motion
    return np.sqrt(positive_motion * np.clip(audio_z, 0.0, None))


def solo_score(motion_z: np.ndarray, audio_z: np.ndarray) -> np.ndarray:
    """The stronger of the two channels taken alone, positive part only.

    This is the "seen OR heard" signal, deliberately blind to whether the
    other channel agreed — that is the whole point of the solo rule.
    """
    positive_motion = np.clip(motion_z, 0.0, None)
    if audio_z.size != motion_z.size:
        return positive_motion
    return np.maximum(positive_motion, np.clip(audio_z, 0.0, None))


def event_score(
    combined: np.ndarray,
    solo: np.ndarray,
    *,
    z_threshold: float = Z_THRESHOLD,
    solo_z_threshold: float = SOLO_Z_THRESHOLD,
) -> np.ndarray:
    """Rank the two detection rules on one comparable scale.

    Each channel is divided by ITS OWN bar and then re-expressed in combined
    units, so a frame sitting exactly on either threshold scores exactly
    ``z_threshold``. That single normalisation is what makes a solo event
    comparable to a combined one without inventing a free weight.

    THIS IS A LIFT, NOT AN IDENTITY. By AM-GM ``solo >= combined`` always
    (the max of two numbers is never below their geometric mean), so with the
    measured bars 2.5 and 5.0 this reduces to ``score = max(combined,
    solo / 2)`` and therefore ``score >= combined`` for EVERY frame, solo
    branch or not. Equality holds exactly when the two channels are balanced
    within 4:1 (``max(m, a) <= 4 * min(m, a)``); past that ratio the solo term
    wins and lifts a combined-detected event above its raw geometric mean.
    Measured on the live 23-clip pool: this moves 4 of 39 events (10%), and on
    clip e3dd74a3 it moves ``payoff_at`` by 2.42s (3.00s instead of 0.58s).

    That lift is deliberate and is the reason the obvious "preserve v1
    ordering" alternative — ``where(combined >= bar, combined, solo / 2)`` —
    was measured and REJECTED. On that same clip it demotes t=3.00
    (motion_z 1.05, audio_z 7.47) below t=0.58 (motion_z 0.42, audio_z 7.02),
    a moment that is strictly louder AND strictly busier, precisely because
    the second channel corroborated it and pushed it onto the combined branch.
    It penalises corroboration; the max() cannot, because:

    - **Floor.** A corroborated event is never scored below what its strongest
      single channel already earned on its own.
    - **Monotone in both channels.** ``combined`` and ``solo`` are each
      non-decreasing in ``motion_z`` and ``audio_z``, and the max of two
      non-decreasing functions is non-decreasing — so raising either channel
      can never lower the score.
    - **No penalty for agreement.** Lifting the weaker channel off zero raises
      ``combined`` and never lowers ``solo``, so evidence from a second
      channel is always non-negative evidence.
    """
    if solo.size != combined.size:
        return np.array(combined, dtype=np.float64)
    normalised = np.maximum(combined / z_threshold, solo / solo_z_threshold)
    return normalised * z_threshold


def pick_events(
    combined: np.ndarray,
    motion_z: np.ndarray,
    audio_z: np.ndarray,
    *,
    fps: float,
    motion_raw: np.ndarray | None = None,
    audio_raw: np.ndarray | None = None,
    z_threshold: float = Z_THRESHOLD,
    solo_z_threshold: float = SOLO_Z_THRESHOLD,
    min_gap_sec: float = MIN_GAP_SEC,
    max_events: int = MAX_EVENTS,
) -> list[dict]:
    """Pick up to ``max_events`` moments that clear ANY detection rule —
    ``combined`` above ``z_threshold`` (seen AND heard), a single channel above
    ``solo_z_threshold`` (seen OR heard), or, when the raw signals are supplied,
    a single channel above ``solo_z_threshold`` measured against a LOCAL spread
    (``zscore_over_local_spread`` — seen OR heard *for this part of the clip*).
    Any peak within ``min_gap_sec`` of a stronger one is suppressed.

    Suppression and the cap run over the UNION, ranked by ``score``, so the
    rules cannot each spend the budget separately.

    ``motion_raw``/``audio_raw`` are the PRE-z-scored channels and are OPTIONAL:
    omit them and this behaves exactly as it did before the local rule existed,
    so every existing caller and every stored annotation keeps its meaning.

    THE LOCAL RULE ONLY ADDS CANDIDATES — it never re-ranks the ones the global
    rules already found. A local-only hit is ranked at exactly ``z_threshold``,
    i.e. at the bar, below every globally-detected event, so ``payoff_at`` can
    never move onto one. That is deliberate: the local estimator is blind to
    how a moment compares to the REST of the clip, which is precisely the
    comparison a payoff has to win.

    MEASURED over the live 23-clip pool (2026-07-20), against the alternatives
    that were tried and rejected for the same operator complaint:

        rule                          pool events   ref clip beats found
        baseline (no local rule)          38          2 of 3  (boarding lost)
        LOCAL SOLO RULE (this one)        39          3 of 3
        drop SOLO_Z_THRESHOLD to 4.5      41          2 of 3  (still lost!)
        drop SOLO_Z_THRESHOLD to 4.0      50          3 of 3
        per-clip adaptive top-3           52          3 of 3
        MAD spread everywhere             70          3 of 3
        trim top 10% from the spread      76          3 of 3
        local spread everywhere           21          1 of 3  (payoffs lost)

    The local rule is the only candidate that buys the operator's moment for
    +1 event across the entire pool; every absolute-bar or global-estimator
    alternative costs 12-38 extra events for the same gain. Per clip, exactly
    ONE clip changed (the reference clip, 2 -> 3 events) and ``payoff_at``
    moved on 0 of 23.

    Returns a NEW list of event dicts sorted by time ascending. Index ``i`` of
    the signals is the transition into frame ``i + 1`` (see
    ``frames.motion_diffs``), so its timestamp is ``(i + 1) / fps``.
    """
    if combined.size == 0 or fps <= 0:
        return []
    solo = solo_score(motion_z, audio_z)
    score = event_score(
        combined, solo, z_threshold=z_threshold, solo_z_threshold=solo_z_threshold
    )
    rank = _rank_with_local_rule(
        score,
        motion_raw,
        audio_raw,
        fps=fps,
        z_threshold=z_threshold,
        solo_z_threshold=solo_z_threshold,
    )
    min_gap_frames = max(int(round(min_gap_sec * fps)), 1)
    kept = _suppress_neighbours(rank, z_threshold, min_gap_frames, max_events)
    events = [
        _event_at(index, combined, motion_z, audio_z, rank, fps=fps)
        for index in sorted(kept)
    ]
    return events


def _rank_with_local_rule(
    score: np.ndarray,
    motion_raw: np.ndarray | None,
    audio_raw: np.ndarray | None,
    *,
    fps: float,
    z_threshold: float,
    solo_z_threshold: float,
) -> np.ndarray:
    """``score``, lifted to exactly ``z_threshold`` wherever the local-spread
    solo rule fires and the global rules did not. Returns a NEW array; returns
    ``score`` unchanged when the raw signals were not supplied.

    Lifting to the bar rather than to the local score is what keeps the local
    rule additive: it makes a moment selectable without letting it outrank a
    moment the whole-clip comparison already endorsed.
    """
    if motion_raw is None or audio_raw is None:
        return score
    if motion_raw.size != score.size or audio_raw.size != score.size:
        return score
    local_solo = solo_score(
        zscore_over_local_spread(motion_raw, fps=fps),
        zscore_over_local_spread(audio_raw, fps=fps),
    )
    local_only = (local_solo >= solo_z_threshold) & (score < z_threshold)
    return np.where(local_only, z_threshold, score)


def _suppress_neighbours(
    score: np.ndarray, z_threshold: float, min_gap_frames: int, max_events: int
) -> list[int]:
    """Greedy non-maximum suppression: walk candidates strongest-first, keep
    one only when no already-kept peak sits within ``min_gap_frames``."""
    candidates = [
        int(index) for index in np.argsort(score)[::-1] if score[index] >= z_threshold
    ]
    kept: list[int] = []
    for index in candidates:
        if len(kept) >= max_events:
            break
        if any(abs(index - chosen) < min_gap_frames for chosen in kept):
            continue
        kept.append(index)
    return kept


def _event_at(
    index: int,
    combined: np.ndarray,
    motion_z: np.ndarray,
    audio_z: np.ndarray,
    score: np.ndarray,
    *,
    fps: float,
) -> dict:
    """Build one event dict. ``kind`` is decided by the MOTION score alone: a
    frame-wide content swap is a cut whether or not it happens to be loud —
    and whether it was found by the combined rule or the solo one, so the solo
    rule can never smuggle a seam in as a payoff candidate.

    ``combined`` stays the honest raw geometric mean (near zero for a solo
    event); ``score`` is the normalised value that actually caused selection.
    """
    motion = float(motion_z[index]) if index < motion_z.size else 0.0
    audio = float(audio_z[index]) if index < audio_z.size else 0.0
    kind = EVENT_KIND_CUT if motion > CUT_Z_THRESHOLD else EVENT_KIND_ACTION
    return {
        "t": round((index + 1) / fps, 2),
        "motion_z": round(motion, 2),
        "audio_z": round(audio, 2),
        "combined": round(float(combined[index]), 2),
        "score": round(float(score[index]), 2),
        "kind": kind,
    }


def classify(events: list[dict]) -> tuple[str, float | None, list[float]]:
    """Summarise a picked event list as ``(clip_type, payoff_at, cuts)``.

    ``payoff_at`` is the timestamp of the strongest ACTION — a cut is the
    editor's seam, never the clip's punchline, so cuts are excluded from the
    choice while still being reported in ``cuts``. A clip whose only signal is
    seams (or which has no events at all) is a ``mood`` clip: real footage with
    no single moment, useful as a breather rather than a hook.

    Ranking is by ``score``, not by ``combined``: a solo-detected event has a
    ``combined`` near zero by construction, so ranking on the raw value would
    make it detectable but never selectable.
    """
    cuts = [event["t"] for event in events if event["kind"] == EVENT_KIND_CUT]
    actions = [event for event in events if event["kind"] == EVENT_KIND_ACTION]
    if not actions:
        return CLIP_TYPE_MOOD, None, cuts
    payoff = max(actions, key=lambda event: event["score"])
    return CLIP_TYPE_EVENT, payoff["t"], cuts
