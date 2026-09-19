"""Optional action classifier: local/LAN model → one label + probability per risky
tool call. Can only escalate (note / ask / deny), never grant. docs/classify.md."""
from agent.classify.command import run_classify_command, startup_warning
from agent.classify.guard import guard_tool_call

__all__ = ["guard_tool_call", "run_classify_command", "startup_warning"]
