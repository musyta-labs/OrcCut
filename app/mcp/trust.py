"""Trust tiers for the MCP tool dictionary (Gate 1 Step 11).

Decision R7: the trust boundary runs THROUGH the tool dictionary, not between
two applications. A tool carries its tier as a declared property here, and the
public dictionary is built by a FILTER that keeps tier-1 tools — never by
subtracting a blocklist from the full set. The difference matters: a new tool
is public only if someone lists it as public, so forgetting to classify it
fails closed (and fails a test, ``tests/test_trust_tiers.py``), rather than
leaking a fresh operator-grade primitive the day it is added.

Two tiers:

* **PUBLIC** — the whole editor vocabulary: project, timeline, trims, texts,
  music, transitions, validation, export, preview, annotations, history. Safe
  to expose to any authenticated caller.
* **OPERATOR** — tools that hand the caller a primitive into this container's
  disk or internal network:

  - ``editor_analyze_media(source=...)`` — reads any local file and, for a URL,
    drives yt-dlp's generic extractor (an SSRF fetch); decoded frames come back
    to the caller.
  - ``editor_add_media(source=...)`` — puts an arbitrary path into the DB as an
    asset, later served from outside ``media_dir``. In Gate 1 the public tier
    registers media by DIRECT UPLOAD only (an HTTP route, not an MCP tool), so
    the whole tool is operator-only here.

    Hiding the TOOL turned out not to cover the PRIMITIVE: ``editor_add_music``
    is public tier and takes the same ``source``, so a public caller could
    register a path through it and read the file back from the asset stream
    route (found 2026-07-30, cross-tenant). Tiers gate whole tools; that hole
    needed an argument-level rule, which now lives in
    ``app.media.source_guard`` and is enforced for non-operator callers in
    ``app.mcp.server._refuse_untrusted_local_path``. A new public tool taking a
    path-shaped argument must be listed there — the tier table alone will not
    catch it.
  - ``editor_restore_snapshot`` — writes an arbitrary project blob as the next
    version, bypassing every mutation guard.

  …AND, since Ворота 4, one tool that is not a container primitive at all:

  - ``editor_delete_project`` — permanently deletes a project, its journal, its
    annotations, its whole version history and its artifacts.

These are NOT "checked more strictly" for the public tier — they are ABSENT
from it. ``tools/list`` under a non-operator caller does not name them, and a
direct call to one is refused.

**WHY DELETION JOINS THEM, AND WHY THAT WIDENS THE TIER'S MEANING ON PURPOSE.**
There was no existing convention to follow: this split has never been about how
DESTRUCTIVE a tool is. ``editor_delete_element`` and ``editor_remove_track``
throw work away and are PUBLIC, and the version-history note below argues a
tool into the public tier precisely by showing it is not a primitive. By that
test alone ``editor_delete_project`` is public — it takes a project id, touches
only what the caller already owns, and names no path and no URL.

It is OPERATOR anyway, for a property no other tool in this dictionary has: it
is the only one whose damage OUTLIVES the mechanism that makes every other
public mutation survivable. The reason a mangled timeline is an acceptable risk
to hand an autonomous agent is ``project_checkpoints`` — the server keeps the
states the project passed through, and ``editor_restore_version`` walks it
back. Deleting the project deletes that history with it. A prompt-injected
agent holding a public token can, with every other tool in the set, cost its
tenant an edit; with this one it costs them the work.

The price of being wrong in this direction is close to nothing, which is what
settles it. The HUMAN surface does not go through MCP at all — the SPA calls
``DELETE /api/projects/{id}``, which is session-authenticated and unaffected —
and a self-hosted deployment with ``MCP_AUTH_ENABLED=false`` is already an
operator by ``caller_is_operator`` below, so the local workflow keeps the tool.
What actually changes is that a multi-tenant deployment must MINT a token
carrying ``editor:operator`` before an agent can delete anything, i.e. deletion
becomes a deliberate grant rather than an ambient capability. That is the same
fail-closed default this whole module exists to express.

**URL import stays out of the public tier entirely** (it is not even a tool
yet): it needs an egress allowlist, deferred to Gate 2, and handing the public
an un-allowlisted internal-network request primitive is the exact hole the
operator tier exists to contain.

**Decision R8, recorded here because this is where the tiers live:**
An external annotation client talks to this server on the
OPERATOR tier and never receives a user's token — it is operator
infrastructure, not a tenant.
"""
from __future__ import annotations

import enum

from app.auth.principal import Principal
from app.config import get_settings

# The scope an authenticated token must carry to reach OPERATOR tools. Added to
# ``KNOWN_SCOPES`` (``app.db.repositories.tokens``) so it can actually be minted;
# no token carries it today, which is the correct default — operator tools are
# invisible until a token is deliberately minted to reach them.
OPERATOR_SCOPE = "editor:operator"


class Tier(enum.StrEnum):
    PUBLIC = "public"
    OPERATOR = "operator"


# EVERY registered MCP tool, classified. This mapping is the single source of
# truth; ``tests/test_trust_tiers.py`` asserts it names exactly the set of
# ``@mcp_server.tool()`` wrappers, so a new tool with no entry fails the build
# rather than defaulting to either tier.
TOOL_TIERS: dict[str, Tier] = {
    # --- OPERATOR: arbitrary path / URL primitives -------------------------
    "editor_add_media": Tier.OPERATOR,
    "editor_analyze_media": Tier.OPERATOR,
    "editor_restore_snapshot": Tier.OPERATOR,
    # --- OPERATOR: irreversible past the version history -------------------
    # The one entry here that is not a container primitive. See the module
    # docstring for the argument — in short, every other public mutation is
    # survivable because ``editor_restore_version`` can walk it back, and this
    # is the tool that deletes the history that makes that true.
    "editor_delete_project": Tier.OPERATOR,
    # --- PUBLIC: the editor vocabulary -------------------------------------
    "editor_create_project": Tier.PUBLIC,
    "editor_get_project": Tier.PUBLIC,
    "editor_list_projects": Tier.PUBLIC,
    "editor_add_clip": Tier.PUBLIC,
    "editor_trim_clip": Tier.PUBLIC,
    "editor_resize_clip": Tier.PUBLIC,
    "editor_split_clip": Tier.PUBLIC,
    "editor_move_clip": Tier.PUBLIC,
    "editor_add_video_track": Tier.PUBLIC,
    "editor_add_audio_track": Tier.PUBLIC,
    "editor_remove_track": Tier.PUBLIC,
    "editor_delete_element": Tier.PUBLIC,
    "editor_set_transform": Tier.PUBLIC,
    "editor_set_clip_audio": Tier.PUBLIC,
    "editor_set_audio_envelope": Tier.PUBLIC,
    "editor_set_fit": Tier.PUBLIC,
    "editor_set_cover": Tier.PUBLIC,
    "editor_auto_captions": Tier.PUBLIC,
    "editor_generate_voiceover": Tier.PUBLIC,
    "editor_add_text": Tier.PUBLIC,
    "editor_update_text": Tier.PUBLIC,
    "editor_update_texts": Tier.PUBLIC,
    "editor_add_music": Tier.PUBLIC,
    "editor_add_transition": Tier.PUBLIC,
    "editor_remove_transition": Tier.PUBLIC,
    "editor_set_keyframes": Tier.PUBLIC,
    "editor_set_color": Tier.PUBLIC,
    "editor_add_overlay": Tier.PUBLIC,
    "editor_update_overlay": Tier.PUBLIC,
    "editor_render_preview": Tier.PUBLIC,
    "editor_validate": Tier.PUBLIC,
    "editor_export": Tier.PUBLIC,
    "editor_reopen_project": Tier.PUBLIC,
    "editor_get_history": Tier.PUBLIC,
    # The version-history trio is PUBLIC even though ``editor_restore_snapshot``
    # is OPERATOR, and the difference is the whole point of that tier. The
    # operator tool takes an ARBITRARY BLOB from the caller and writes it as the
    # project, bypassing every mutation guard. These three take a project id and
    # an integer, and can only ever read or replay a state THIS SERVER itself
    # recorded for a project the caller already owns — no arbitrary document, no
    # path, no URL. Nothing here is a primitive into the container.
    "editor_save_version": Tier.PUBLIC,
    "editor_list_versions": Tier.PUBLIC,
    "editor_restore_version": Tier.PUBLIC,
    "editor_add_annotation": Tier.PUBLIC,
    "editor_list_annotations": Tier.PUBLIC,
    "editor_resolve_annotation": Tier.PUBLIC,
}


def caller_is_operator(principal: Principal | None) -> bool:
    """Whether the current caller may reach OPERATOR tools.

    Two ways to be an operator, and the first is a deployment mode, not a
    bypass:

    * **Auth disabled** (``MCP_AUTH_ENABLED=false``) — the deployment is
      single-tenant and the sole caller IS the operator. Hiding the
      path/URL tools here would break the self-hosted local workflow (adding
      media by ``source`` over MCP is how it is done without a browser), and
      would mean the tier code shipping to production is not the one exercised
      locally. ``principal`` is ``None`` on this path because there is no token
      to resolve.
    * **Auth enabled** — the token must carry ``OPERATOR_SCOPE``. No token does
      by default, so operator tools stay invisible until one is deliberately
      minted to reach them.
    """
    if not get_settings().mcp_auth_enabled:
        return True
    return principal is not None and OPERATOR_SCOPE in principal.scopes


def visible_tool_names(all_names: list[str], principal: Principal | None) -> set[str]:
    """The subset of ``all_names`` this caller may see in ``tools/list``.

    An operator sees everything registered. A public caller sees only tools
    explicitly marked ``PUBLIC`` — a name missing from ``TOOL_TIERS`` is
    treated as operator-only (hidden), so an unclassified tool fails closed
    rather than leaking. Classification drift is caught loudly by the test, but
    the runtime default is still the safe one.
    """
    if caller_is_operator(principal):
        return set(all_names)
    return {
        name for name in all_names
        if TOOL_TIERS.get(name, Tier.OPERATOR) is Tier.PUBLIC
    }
