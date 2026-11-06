# SafeJEV — notes

Running log of decisions and deviations from the design doc.

## 2026-11-06

- Own git repo in `code_repo/safejev/`; added `safejev/` to the outer nano-jev repo's
  `.git/info/exclude` (same as tinyjev / microjev / cyber_jev).
- **jev-core question (§3, §15 open):** not extracted yet. SafeJEV keeps its shared pieces
  (calibration, decision schema, audit format) in `safejev/core/` with no import from
  cyberjev, so they can be lifted into a `jev-core` package later without changes.
