## Purpose

This local Claude configuration grants a narrow set of permissions for work involving a Somm-specific CEO plan. It allows creating a local plan directory, copying one generated artifact into it, and reading that project directory.

## How it works

The file defines a `permissions.allow` list containing three exact capability rules (`.claude/settings.local.json:2`). Two rules authorize specific Bash commands: creating `~/.gstack/projects/somm/ceo-plans` and copying a particular tool-result file to a dated Markdown plan (`.claude/settings.local.json:4`, `.claude/settings.local.json:5`). The third permits reads beneath the Somm directory in `~/.gstack/projects` (`.claude/settings.local.json:6`).

No lifecycle or runtime behavior is defined here. How the surrounding Claude tooling loads or merges this local configuration is not established by this file alone.

## Key surfaces

- `mkdir -p ~/.gstack/projects/somm/ceo-plans` — creates the plan destination directory (`.claude/settings.local.json:4`).
- `cp …/bx4ga4jfj.txt …/2026-04-17-codex-ceo-voice.md` — installs one generated result as a named CEO-plan document (`.claude/settings.local.json:5`).
- `Read(//home/marc/.gstack/projects/somm/**)` — permits reading the resulting project-local gstack tree (`.claude/settings.local.json:6`).

## Design decisions

- Bash access is command-specific rather than broadly allowing shell execution, limiting authorization to two known filesystem operations.
- Read access is constrained to the Somm subtree, while the copied source is identified by an exact path. This appears designed for least privilege, although the significance of the doubled leading slash in the read rule is unclear from this file alone.
- The source path contains a session-specific identifier, making this permission tightly coupled to one generated artifact rather than reusable across sessions.

## One-liner

A least-privilege local permission file for placing and reading one Somm CEO-plan artifact.