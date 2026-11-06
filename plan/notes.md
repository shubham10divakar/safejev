# SafeJEV — notes

Running log of decisions and deviations from the design doc.

## 2026-11-06

- Own git repo in `code_repo/safejev/`; added `safejev/` to the outer nano-jev repo's
  `.git/info/exclude` (same as tinyjev / microjev / cyber_jev).
- **jev-core question (§3, §15 open):** not extracted yet. SafeJEV keeps its shared pieces
  (calibration, decision schema, audit format) in `safejev/core/` with no import from
  cyberjev, so they can be lifted into a `jev-core` package later without changes.
- **Heuristic backend (not in the design doc):** no weights exist yet, so every user-facing
  adapter except `agent-trajectory` has a deterministic rule implementation
  (`safejev/detectors/heuristic.py`, `precheck.py`). It is the default (`backend="auto"` uses
  it unless `SAFEJEV_WEIGHTS` is set) and the rule baseline for every experiment. Versions
  carry `+heuristic` so no one mistakes its scores for calibrated ones.
- **Decision head vs. types:** fusion works on per-type scores (one Detection per type). The
  encoder's 3-way decision head is trained and calibrated and shown in the reason text, but the
  policy decides from types + severity. Revisit after the pilot: if the decision head is
  better calibrated than max-type, add a `decision`-typed detection.
- **Severity:** taken from `taxonomy.DEFAULT_SEVERITY` per type; the severity head is trained
  but only used when `use_severity_head=True` in the adapter meta.
- **Policy order:** authz → fail-closed → rules → severity thresholds → require_approval;
  the most restrictive wins.
- **MCP proxy approvals:** stdin/stdout carry MCP traffic, so the proxy cannot prompt on the
  terminal; `--approve-webhook` or deny.
- **Rate limits** are consumed at check time (a blocked-for-other-reasons call does not
  consume one).
- **Converters** for TS-Bench / R-Judge use assumed field names; nothing was downloaded.
- `training/` imports `safejev` and is run from the repo root (`python -m training...`);
  pytest has `pythonpath = ["."]`.
