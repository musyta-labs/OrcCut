---
name: video-editing
description: Edit and render video through the Orccut MCP server — create projects, build timelines from local or URL media, add text/captions/music, validate, and export. Use whenever the user asks to cut, assemble, caption, or render video.
---

# Editing video through Orccut

Works the same against a self-hosted server (`http://127.0.0.1:8100/mcp`) or
the hosted service at `https://orccut.com/mcp` (bearer token, browser editor
on top — the user sees your edits live at orccut.com/projects).

You are connected to a real video editor. It keeps **projects**: immutable
timeline snapshots with a journaled history. Every tool call that edits
returns the **full updated project snapshot** — always work from the latest
returned snapshot, never from memory of an earlier one.

## Conventions that make your edits reliable

- **Errors come back as data, not exceptions**: a failed call returns
  `{"error": "<reason>"}`. Read it, fix the cause, retry deliberately —
  never retry the identical call in a loop.
- **Element ids, media ids, track ids** are in every snapshot. Use ids from
  the latest snapshot; ids never survive a delete.
- **Times are seconds** (floats). Positions like `pos` accept presets
  (`"center"`, `"top"`, `"bottom"`...).
- **One gesture = one tool call.** The journal (`editor_get_history`) records
  each named operation; prefer several precise calls over one clever one.
- Long operations (export, captions, analysis) run synchronously and can take
  minutes on big media — call them once and wait.

## The core workflow

1. **Create**: `editor_create_project` (optionally with `metadata` tags).
2. **Ingest**: `editor_add_media(project_id, source)` — `source` is a local
   path or an http(s) URL (URL import downloads server-side). The returned
   media entry carries duration/resolution once probed.
3. **Assemble**: `editor_add_clip(project_id, media_id, start_time, duration,
   trim_start)` places footage on the timeline; `editor_trim_clip`,
   `editor_split_clip`, `editor_move_clip`, `editor_resize_clip` refine it.
   Extra layers: `editor_add_video_track` / `editor_add_audio_track`.
4. **Dress**: `editor_add_text(project_id, content, start, duration, pos,
   size)` for titles; `editor_add_overlay` for images/GIFs/video overlays;
   `editor_add_music(project_id, source)` for a music bed;
   `editor_set_transform` (+ `editor_set_keyframes` for motion),
   `editor_set_color`, `editor_add_transition`.
5. **Captions** (if the ASR extra is installed): `editor_auto_captions
   (project_id)` transcribes the timeline audio and lays word-timed captions.
   `editor_update_texts` fixes any wording afterwards.
6. **Check before rendering**: `editor_validate(project_id)` reports
   structural problems; `editor_render_preview(project_id, at_time)` returns
   a frame so you can *see* the composition at a timestamp.
7. **Export**: `editor_export(project_id)` renders the final file and
   returns a download path/URL. Exports are swept after a retention window
   (24 h by default) — tell the user to save the file, or re-export.

## Understanding the footage

`editor_analyze_media(project_id, media_id)` returns scenes, audio peaks and
keyframes — use it to pick cut points instead of guessing timestamps.
`editor_get_project` re-reads the current snapshot; `editor_get_history`
shows every operation ever applied (useful to explain what you did).

## Safety nets

- `editor_save_version(project_id, label)` freezes a named version before a
  risky series of edits; `editor_list_versions` / `editor_restore_version`
  roll back.
- Annotations (`editor_add_annotation`) let you or the user pin notes to
  timeline moments; resolve them with `editor_resolve_annotation`.

## A typical request, end to end

> "Make a 30-second cut of talk.mp4 with a title and captions."

```
editor_create_project {}
editor_add_media {project_id, source: "/path/to/talk.mp4"}
editor_analyze_media {project_id, media_id}          # find the good segment
editor_add_clip {project_id, media_id, trim_start: 42.0, duration: 30.0}
editor_add_text {project_id, content: "The Talk", start: 0, duration: 3,
                 pos: "center", size: 72}
editor_auto_captions {project_id}
editor_validate {project_id}
editor_export {project_id}
```

Then hand the user the export path and remind them it is swept after the
retention window.
