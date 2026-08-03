# Skills

Ready-made instructions that teach an AI agent to edit video through this
MCP server. They are plain markdown — nothing here is specific to one agent
product.

- **Claude Code**: copy a skill directory into `.claude/skills/` in your
  project (or `~/.claude/skills/` globally); it will be picked up by name.
- **Claude Desktop / other MCP clients**: paste the skill file's content into
  your system prompt or project instructions.
- **Anything else**: treat `SKILL.md` as reference documentation for the tool
  surface and hand it to your model however your harness passes context.

| Skill | What it teaches |
|---|---|
| [video-editing](video-editing/SKILL.md) | The full workflow: project → media → timeline edits → captions → validate → export, plus the conventions that make agent edits reliable |
