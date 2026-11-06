"""SafeJEV: calibrated local detectors + deterministic governance for LLM apps and agents.

    from safejev import SafeJEV
    sj = SafeJEV(modules=["guard", "agent"])
    result = sj.scan(user_input="Find duplicate files in /documents",
                     agent_action={"tool": "delete_file",
                                   "arguments": {"path": "/documents/report.pdf"}})
    print(result.decision, result.risk_level)
"""

__version__ = "0.1.0.dev0"

from .core.detector import ScanInput  # noqa: E402
from .core.result import Detection, ScanResult  # noqa: E402
from .engine import CallVerdict, SafeJEV, ToolBlocked  # noqa: E402

__all__ = ["SafeJEV", "ScanInput", "ScanResult", "Detection", "ToolBlocked", "CallVerdict",
           "__version__"]
