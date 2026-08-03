"""Structural type check of a reconstructed project (R-04).

``serialization.project_from_dict`` rebuilds the frozen dataclasses from a plain
dict by keyword — and a dataclass enforces nothing at runtime, so a snapshot
whose ``transform.pos_x`` is the string ``"1;movie=/etc/passwd"`` reconstructs
without complaint and lands, unescaped, in an ffmpeg filter expression or an MLT
property string. The undo/redo path (``mutations.restore_snapshot``) accepts
exactly such a snapshot from the client, which makes it the widest of R-04's
vectors: one request carries EVERY numeric field at once.

Describing the snapshot as a schema next to ``serialization`` would duplicate it
and drift from it. This walks the reconstructed object against the dataclasses'
OWN annotations instead, so it cannot fall out of step with the model, and it
guards every surface (HTTP, CLI and MCP) from one place.

Deliberately a check, not a coercion: the editor's immutability contract means
nothing here may hand back a rewritten project.

``int`` is accepted wherever ``float`` is declared — ``JSON.stringify`` renders
``5.0`` as ``5``, so a browser-round-tripped snapshot legitimately carries ints
in float fields, and rejecting those would break undo for every user. What it
does NOT accept is a value that is not a number at all, or one that is not
finite (``json.loads`` parses the non-standard ``Infinity``/``NaN`` literals,
and ``W*inf`` is a filter expression that only burns a render slot).
"""
from __future__ import annotations

import math
import types
from dataclasses import fields, is_dataclass
from typing import Any, Union, get_args, get_origin, get_type_hints


def check_types(value: Any, *, path: str = "project") -> None:
    """Raise ``TypeError`` naming the first field whose value contradicts its
    declared annotation. ``value`` is expected to be a model dataclass."""
    if not is_dataclass(value):
        raise TypeError(f"{path}: expected a dataclass, got {type(value).__name__}")
    _check_dataclass(value, path)


def _check_dataclass(instance: Any, path: str) -> None:
    hints = get_type_hints(type(instance))
    for field in fields(instance):
        _check(getattr(instance, field.name), hints[field.name], f"{path}.{field.name}")


def _check(value: Any, annotation: Any, path: str) -> None:
    if annotation is Any:
        return
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        _check_union(value, get_args(annotation), path)
        return
    if origin is tuple:
        _check_tuple(value, get_args(annotation), path)
        return
    if annotation is type(None):
        if value is not None:
            raise TypeError(f"{path}: expected null, got {type(value).__name__}")
        return
    if is_dataclass(annotation):
        if not isinstance(value, annotation):
            raise TypeError(
                f"{path}: expected {annotation.__name__}, got {type(value).__name__}"
            )
        _check_dataclass(value, path)
        return
    _check_scalar(value, annotation, path)


def _check_union(value: Any, members: tuple[Any, ...], path: str) -> None:
    for member in members:
        try:
            _check(value, member, path)
        except TypeError:
            continue
        return
    # The offending value is attacker-chosen text and is never quoted back —
    # this message reaches the client as "invalid snapshot: <this>".
    raise TypeError(f"{path}: {type(value).__name__} matches none of its allowed types")


def _check_tuple(value: Any, args: tuple[Any, ...], path: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{path}: expected a tuple, got {type(value).__name__}")
    # ``tuple[X, ...]`` (homogeneous, the shape every collection field uses)
    # versus a fixed-length ``tuple[str, str]`` (metadata pairs).
    if len(args) == 2 and args[1] is Ellipsis:
        for index, item in enumerate(value):
            _check(item, args[0], f"{path}[{index}]")
        return
    if len(value) != len(args):
        raise TypeError(f"{path}: expected {len(args)} items, got {len(value)}")
    for index, (item, arg) in enumerate(zip(value, args, strict=True)):
        _check(item, arg, f"{path}[{index}]")


def _check_scalar(value: Any, annotation: Any, path: str) -> None:
    # ``bool`` is a subclass of ``int``: a stray ``true`` in a numeric field
    # would otherwise pass and render as ``1``.
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{path}: expected a boolean, got {type(value).__name__}")
        return
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{path}: expected an integer, got {type(value).__name__}")
        return
    if annotation is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{path}: expected a number, got {type(value).__name__}")
        if not math.isfinite(value):
            raise TypeError(f"{path}: expected a finite number")
        return
    if not isinstance(value, annotation):
        raise TypeError(
            f"{path}: expected {getattr(annotation, '__name__', annotation)}, "
            f"got {type(value).__name__}"
        )
