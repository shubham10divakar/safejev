"""Guard an agent's tools: every call is screened before it runs."""
from safejev import SafeJEV, ToolBlocked
from safejev.governance import CLIApprover, Policy

policy = Policy(
    block_on=["critical"],
    ask_on=["high"],
    require_approval_tools=["delete_file", "send_email", "transfer_funds", "external_upload"],
    fail_closed=True,
    audit_log="./safejev_audit.jsonl",
)
sj = SafeJEV(modules=["agent", "guard"], policy=policy, backend="heuristic",
             approver=CLIApprover(timeout=30))


@sj.guard_tool
def list_files(path: str):
    return ["report.pdf", "report (1).pdf"]


@sj.guard_tool(sensitive=True)
def delete_file(path: str):
    return f"deleted {path}"


with sj.session(user_goal="Find duplicate files in /documents"):
    print(list_files("/documents"))
    try:
        print(delete_file("/documents/report.pdf"))     # asks you on the terminal
    except ToolBlocked as e:
        print("blocked:", e)

print("\naudit log: safejev_audit.jsonl  (summarise with: safejev report)")
