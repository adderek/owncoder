"""Core agent runtime: turn loop, streaming, tool dispatch, model routing.

Deliberately empty of re-exports — every caller in this codebase imports
directly from the submodule it needs (``agent.core.turn``, ``agent.core.agent``,
…). Re-exporting here would make importing *any* single submodule (e.g.
``agent.core.llm_retry``) eagerly load the whole runtime (Agent, turn, streaming,
tool_calls, …), which is both slower and needlessly couples lightweight,
independent modules to the full agent stack.
"""
