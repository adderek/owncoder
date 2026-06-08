from dataclasses import dataclass, field
from typing import List, Dict, Set, Any, Optional

@dataclass
class ProtectedEntity:
    term: str
    entity_type: str  # e.g., "file", "function", "error_code"

@dataclass
class MemorySegment:
    segment_id: int
    raw_turns: List[Dict[str, Any]]
    local_summary: Optional[str] = None
    protected_entities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)