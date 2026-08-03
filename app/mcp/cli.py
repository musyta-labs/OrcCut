"""CLI twin of the editor_* MCP tools.

Lets a session drive the agent video editor deterministically, without
waiting for the MCP server to be (re)connected — new MCP tools only appear
after the server restarts:

    .venv/bin/python -m app.mcp.cli [--email you@example.com] payload.json

payload.json is either a single op::

    {"op": "add_clip", "project_id": "abc", "args": {"media_id": "..."}}

or a batch::

    {"ops": [
        {"op": "create_project", "args": {"metadata": {"client": "demo"}}},
        {"op": "add_media", "project_id": "abc", "args": {"source": "/path/clip.mp4"}}
    ]}

``project_id`` (when present) is merged into ``args`` before dispatch, so it can
sit at the op's top level for readability. Each op maps to the same-named
``app.mcp.tools`` logic function. Prints the result(s) as JSON; exit code is 1
when any result is an error.

**A batch is NOT a transaction.** All ops share one session, but exit code 1
does not mean "nothing happened": an op that fails by returning an error dict
stops the batch, and everything before it still commits when the session
scope exits. ``export`` goes further — it commits the session MID-op (its
status changes must be durable before/after the render, see
``tools.export``'s transaction-discipline note), making every prior op in the
batch durable at that point even if a later op raises. Treat each op as
individually durable; don't build workflows that rely on a failed batch
rolling back.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import select

from app.auth.principal import Principal, bind_principal
from app.db.base import session_scope
from app.db.models import UserRow, UserStatus
from app.db.repositories import users as users_repo
from app.mcp import argspec, tools

# Only these logic functions are dispatchable from the CLI (no server internals).
ALLOWED_OPS = {
    "create_project",
    "get_project",
    "add_media",
    "add_video_track",
    "add_audio_track",
    "remove_track",
    "add_clip",
    "trim_clip",
    "resize_clip",
    "split_clip",
    "cut_range",
    "move_clip",
    "delete_element",
    "set_transform",
    "set_clip_audio",
    "set_audio_envelope",
    "set_fit",
    "set_cover",
    "auto_captions",
    "generate_voiceover",
    "add_text",
    "update_text",
    "update_texts",
    "update_overlay",
    "add_music",
    "add_transition",
    "remove_transition",
    "set_keyframes",
    "set_color",
    "add_overlay",
    "restore_snapshot",
    "analyze_media",
    "render_preview",
    "validate",
    "export",
    "reopen_project",
    "list_projects_tool",
    "get_history_tool",
}


def load_payload(path: str) -> tuple[dict | None, str | None]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"cannot read payload: {exc}"
    if not isinstance(payload, dict):
        return None, "payload must be a JSON object"
    if "ops" not in payload and "op" not in payload:
        return None, "payload must have an 'op' or an 'ops' list"
    return payload, None


def _normalize_ops(payload: dict) -> list[dict]:
    return payload["ops"] if "ops" in payload else [payload]


def _run_op(session, op: dict):
    name = op.get("op")
    if name not in ALLOWED_OPS:
        return {"error": f"unknown op: {name!r}"}
    args = dict(op.get("args") or {})
    if "project_id" in op:
        args.setdefault("project_id", op["project_id"])
    func = getattr(tools, name)
    # The payload is a file of raw JSON, exactly as untrusted as an HTTP body:
    # values are type-checked against the tool's signature before the call, or
    # a string in a numeric field reaches the render DSL unescaped (R-04).
    try:
        args = argspec.validate_args(name, args)
    except argspec.ArgValidationError as exc:
        return {"error": f"invalid args for {name!r}: {exc}"}
    try:
        return func(session, **args)
    except TypeError as exc:  # bad/missing args for the op
        return {"error": f"invalid args for {name!r}: {exc}"}


def _is_error(result) -> bool:
    return isinstance(result, dict) and "error" in result


def _resolve_owner(session, email: str | None) -> tuple[int | None, str | None]:
    """``(user_id, error)`` — the account this CLI run acts as.

    Gate 1 Step 7: every op now writes or reads rows that belong to somebody,
    so the CLI has to name an account. ``--email`` is explicit and mirrors
    ``app.auth.admin_cli``'s existing convention.

    Without ``--email`` it defaults to the only account there is — and REFUSES
    when there is more than one, rather than picking the first. A default that
    silently chose an owner is precisely the failure this step exists to
    prevent: it would file a project under one tenant while the operator
    believed they were working as another, and nothing would look wrong until
    the rows were missing.
    """
    if email is not None:
        user = users_repo.find_by_email(session, email)
        if user is None:
            return None, f"no account for {email!r}"
        if user.status != UserStatus.ACTIVE.value:
            return None, f"account {email!r} is {user.status}"
        return user.id, None

    active = session.execute(
        select(UserRow.id, UserRow.email).where(UserRow.status == UserStatus.ACTIVE.value)
    ).all()
    if not active:
        # Auth off means single-tenant — the same reading app.mcp.context
        # applies to /mcp — so a fresh database mints the bootstrap owner
        # rather than refusing. With auth ON the deployment is multi-tenant
        # and inventing an owner is exactly what this function must never do.
        from app.config import get_settings

        if not get_settings().mcp_auth_enabled:
            return users_repo.ensure_bootstrap_user(session), None
        return None, (
            "no accounts exist — with MCP_AUTH_ENABLED=false the CLI creates the "
            "single owner itself; in a multi-tenant deployment create an account "
            "with the operator CLI"
        )
    if len(active) > 1:
        names = ", ".join(sorted(email for _, email in active))
        return None, f"several accounts exist ({names}); pass --email to choose one"
    return active[0][0], None


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    email: str | None = None
    if "--email" in args:
        index = args.index("--email")
        if index + 1 >= len(args):
            print(json.dumps({"error": "--email requires a value"}))
            return 1
        email = args[index + 1]
        args = args[:index] + args[index + 2:]
    if len(args) != 1:
        print(json.dumps({"error": "usage: python -m app.mcp.cli [--email ADDR] <payload.json>"}))
        return 1
    payload, error = load_payload(args[0])
    if payload is None:
        print(json.dumps({"error": error}))
        return 1
    results: list = []
    failed = False
    with session_scope() as session:
        owner_id, owner_error = _resolve_owner(session, email)
        if owner_id is None:
            print(json.dumps({"error": owner_error}))
            return 1
        with bind_principal(Principal(user_id=owner_id, token_id=None, scopes=())):
            for op in _normalize_ops(payload):
                result = _run_op(session, op)
                results.append(result)
                if _is_error(result):
                    failed = True
                    break
    output = results[0] if "op" in payload else results
    print(json.dumps(output, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
