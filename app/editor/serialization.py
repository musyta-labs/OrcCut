"""Lossless JSON-safe (de)serialization for ``EditorProject``.

``project_to_dict`` / ``project_from_dict`` round-trip a project through plain
dict/list/str/int/float/bool/None so it can be stored as JSON. The tagged
element union is discriminated by the ``kind`` field.
"""
from __future__ import annotations

from app.editor.errors import EditorError
from app.editor.model import (
    AudioElement,
    ClipElement,
    ColorAdjust,
    Element,
    EditorProject,
    LoudnessTarget,
    MediaAsset,
    OverlayElement,
    TextElement,
    TextStyle,
    Track,
    Transform,
    TransformKeyframe,
    TransitionSpec,
    VolumeKeyframe,
    metadata_from_dict,
    metadata_to_dict,
)


def _transform_to_dict(transform: Transform) -> dict:
    return {
        "scale": transform.scale,
        "pos_x": transform.pos_x,
        "pos_y": transform.pos_y,
        "speed": transform.speed,
        "crop_zoom": transform.crop_zoom,
    }


def _transform_from_dict(data: dict) -> Transform:
    return Transform(
        scale=data["scale"],
        pos_x=data["pos_x"],
        pos_y=data["pos_y"],
        speed=data["speed"],
        crop_zoom=data["crop_zoom"],
    )


def _transition_to_dict(transition: TransitionSpec | None) -> dict | None:
    if transition is None:
        return None
    return {"kind": transition.kind, "duration": transition.duration}


def _transition_from_dict(data: dict | None) -> TransitionSpec | None:
    if data is None:
        return None
    return TransitionSpec(kind=data["kind"], duration=data["duration"])


def _keyframe_to_dict(kf: TransformKeyframe) -> dict:
    return {
        "time": kf.time,
        "scale": kf.scale,
        "pos_x": kf.pos_x,
        "pos_y": kf.pos_y,
        "opacity": kf.opacity,
        "rotation": kf.rotation,
    }


def _keyframe_from_dict(data: dict) -> TransformKeyframe:
    return TransformKeyframe(
        time=data["time"],
        scale=data["scale"],
        pos_x=data["pos_x"],
        pos_y=data["pos_y"],
        opacity=data["opacity"],
        rotation=data["rotation"],
    )


def _color_to_dict(color: ColorAdjust | None) -> dict | None:
    if color is None:
        return None
    return {
        "brightness": color.brightness,
        "contrast": color.contrast,
        "saturation": color.saturation,
        "gamma": color.gamma,
    }


def _color_from_dict(data: dict | None) -> ColorAdjust | None:
    if data is None:
        return None
    return ColorAdjust(
        brightness=data["brightness"],
        contrast=data["contrast"],
        saturation=data["saturation"],
        gamma=data["gamma"],
    )
def _volume_keyframes_to_list(keyframes: tuple[VolumeKeyframe, ...]) -> list[dict]:
    return [{"time": kf.time, "volume": kf.volume} for kf in keyframes]


def _volume_keyframes_from_list(data: list[dict] | None) -> tuple[VolumeKeyframe, ...]:
    """Absent/None = a project persisted before envelopes existed: no keyframes."""
    return tuple(VolumeKeyframe(time=kf["time"], volume=kf["volume"]) for kf in (data or []))


def _loudness_to_dict(loudness: LoudnessTarget | None) -> dict | None:
    if loudness is None:
        return None
    return {"i": loudness.i, "lra": loudness.lra, "tp": loudness.tp}


def _loudness_from_dict(data: dict | None) -> LoudnessTarget | None:
    """``None`` = normalization deliberately disabled; a MISSING key = a project
    persisted before loudness existed, which adopts the default target."""
    if data is None:
        return None
    return LoudnessTarget(i=data["i"], lra=data["lra"], tp=data["tp"])


def _style_to_dict(style: TextStyle) -> dict:
    return {
        "font_path": style.font_path,
        "font_id": style.font_id,
        "size": style.size,
        "color": style.color,
        "pos": style.pos,
        "pos_x": style.pos_x,
        "pos_y": style.pos_y,
    }


def _style_from_dict(data: dict) -> TextStyle:
    return TextStyle(
        font_path=data["font_path"],
        size=data["size"],
        color=data["color"],
        pos=data["pos"],
        # Absent keys = style persisted before explicit offsets existed: fall
        # back to the named preset rather than KeyError on rows already in the DB.
        pos_x=data.get("pos_x"),
        pos_y=data.get("pos_y"),
        # Absent = a style persisted before fonts became library assets (R-33),
        # which is exactly what ``None`` means going forward: the deployment's
        # configured font. That equivalence is why no snapshot in the database
        # needs rewriting — the stored ``font_path`` beside it is re-resolved
        # from this id on every ingress and before every render, so a stale
        # value can neither be trusted nor break the project.
        font_id=data.get("font_id"),
    )


def _element_to_dict(element: Element) -> dict:
    if isinstance(element, ClipElement):
        return {
            "id": element.id,
            "kind": "clip",
            "media_id": element.media_id,
            "start_time": element.start_time,
            "duration": element.duration,
            "trim_start": element.trim_start,
            "trim_end": element.trim_end,
            "transform": _transform_to_dict(element.transform),
            "volume": element.volume,
            "muted": element.muted,
            "opacity": element.opacity,
            "transition_in": _transition_to_dict(element.transition_in),
            "keyframes": [_keyframe_to_dict(kf) for kf in element.keyframes],
            "color": _color_to_dict(element.color),
            "fit": element.fit,
            "fade_in_sec": element.fade_in_sec,
            "fade_out_sec": element.fade_out_sec,
            "volume_keyframes": _volume_keyframes_to_list(element.volume_keyframes),
        }
    if isinstance(element, TextElement):
        return {
            "id": element.id,
            "kind": "text",
            "content": element.content,
            "start_time": element.start_time,
            "duration": element.duration,
            "style": _style_to_dict(element.style),
            "role": element.role,
        }
    if isinstance(element, AudioElement):
        return {
            "id": element.id,
            "kind": "audio",
            "media_id": element.media_id,
            "start_time": element.start_time,
            "duration": element.duration,
            "volume": element.volume,
            "gain_db": element.gain_db,
            "fade_in_sec": element.fade_in_sec,
            "fade_out_sec": element.fade_out_sec,
            "volume_keyframes": _volume_keyframes_to_list(element.volume_keyframes),
        }
    if isinstance(element, OverlayElement):
        return {
            "id": element.id,
            "kind": "overlay",
            "media_id": element.media_id,
            "start_time": element.start_time,
            "duration": element.duration,
            "x": element.x,
            "y": element.y,
            "w": element.w,
            "h": element.h,
            "opacity": element.opacity,
        }
    raise EditorError(f"Cannot serialize unknown element type: {type(element)!r}")


def _element_from_dict(data: dict) -> Element:
    kind = data.get("kind")
    if kind == "clip":
        return ClipElement(
            id=data["id"],
            media_id=data["media_id"],
            start_time=data["start_time"],
            duration=data["duration"],
            trim_start=data["trim_start"],
            trim_end=data["trim_end"],
            transform=_transform_from_dict(data["transform"]),
            volume=data["volume"],
            muted=data["muted"],
            # Absent key = a clip persisted before the opacity field existed:
            # fall back to fully opaque rather than KeyError on old rows.
            opacity=data.get("opacity", 1.0),
            # Absent keys = a clip persisted before phase 4: fall back to the
            # neutral "no transition/keyframes/color" defaults rather than
            # KeyError on rows already in the DB.
            transition_in=_transition_from_dict(data.get("transition_in")),
            keyframes=tuple(_keyframe_from_dict(kf) for kf in data.get("keyframes", ())),
            color=_color_from_dict(data.get("color")),
            # Absent keys = a clip persisted before fit/envelopes existed:
            # fall back to the neutral defaults rather than KeyError.
            fit=data.get("fit", "cover"),
            fade_in_sec=data.get("fade_in_sec", 0.0),
            fade_out_sec=data.get("fade_out_sec", 0.0),
            volume_keyframes=_volume_keyframes_from_list(data.get("volume_keyframes")),
        )
    if kind == "text":
        return TextElement(
            id=data["id"],
            content=data["content"],
            start_time=data["start_time"],
            duration=data["duration"],
            style=_style_from_dict(data["style"]),
            # Absent key = a text element persisted before the role marker
            # existed: fall back to unmarked rather than KeyError on old rows.
            role=data.get("role"),
        )
    if kind == "audio":
        return AudioElement(
            id=data["id"],
            media_id=data["media_id"],
            start_time=data["start_time"],
            duration=data["duration"],
            volume=data["volume"],
            gain_db=data["gain_db"],
            # Absent keys = an audio bed persisted before envelopes existed.
            fade_in_sec=data.get("fade_in_sec", 0.0),
            fade_out_sec=data.get("fade_out_sec", 0.0),
            volume_keyframes=_volume_keyframes_from_list(data.get("volume_keyframes")),
        )
    if kind == "overlay":
        return OverlayElement(
            id=data["id"],
            media_id=data["media_id"],
            start_time=data["start_time"],
            duration=data["duration"],
            x=data["x"],
            y=data["y"],
            w=data["w"],
            h=data["h"],
            opacity=data["opacity"],
        )
    raise EditorError(f"Cannot deserialize unknown element kind: {kind!r}")


def _track_to_dict(track: Track) -> dict:
    return {
        "id": track.id,
        "type": track.type,
        "elements": [_element_to_dict(element) for element in track.elements],
    }


def _track_from_dict(data: dict) -> Track:
    return Track(
        id=data["id"],
        type=data["type"],
        elements=tuple(_element_from_dict(el) for el in data["elements"]),
    )


def _asset_to_dict(asset: MediaAsset) -> dict:
    return {
        "id": asset.id,
        "source": asset.source,
        "duration_sec": asset.duration_sec,
        "local_path": asset.local_path,
    }


def _asset_from_dict(data: dict) -> MediaAsset:
    return MediaAsset(
        id=data["id"],
        source=data["source"],
        duration_sec=data["duration_sec"],
        local_path=data["local_path"],
    )


def project_to_dict(project: EditorProject) -> dict:
    """Serialize a project to a JSON-safe dict (lossless round-trip)."""
    return {
        "id": project.id,
        "metadata": metadata_to_dict(project.metadata),
        "version": project.version,
        "aspect_w": project.aspect_w,
        "aspect_h": project.aspect_h,
        "fps": project.fps,
        "target_sec": project.target_sec,
        "min_clips": project.min_clips,
        "max_clips": project.max_clips,
        "min_duration_sec": project.min_duration_sec,
        "max_duration_sec": project.max_duration_sec,
        "tracks": [_track_to_dict(track) for track in project.tracks],
        "assets": [_asset_to_dict(asset) for asset in project.assets],
        "loudness": _loudness_to_dict(project.loudness),
        "cover_time": project.cover_time,
    }


def project_from_dict(data: dict) -> EditorProject:
    """Reconstruct a project from a dict produced by ``project_to_dict``."""
    return EditorProject(
        id=data["id"],
        metadata=metadata_from_dict(data.get("metadata")),
        version=data["version"],
        aspect_w=data["aspect_w"],
        aspect_h=data["aspect_h"],
        fps=data["fps"],
        target_sec=data["target_sec"],
        # Absent keys = a project persisted before these fields existed: fall
        # back to the neutral defaults rather than KeyError on old rows.
        min_clips=data.get("min_clips", 1),
        max_clips=data.get("max_clips"),
        min_duration_sec=data.get("min_duration_sec", 0.0),
        max_duration_sec=data.get("max_duration_sec"),
        tracks=tuple(_track_from_dict(t) for t in data["tracks"]),
        assets=tuple(_asset_from_dict(a) for a in data["assets"]),
        # Absent key = pre-loudness project: fall back to the default target
        # rather than KeyError on rows already in the DB.
        loudness=(
            _loudness_from_dict(data["loudness"])
            if "loudness" in data
            else LoudnessTarget()
        ),
        # Absent key = a project persisted before cover frames existed: no
        # cover requested, same as the model default.
        cover_time=data.get("cover_time"),
    )
