from dataclasses import dataclass, field
from typing import List


@dataclass
class CompactionRecord:
    session_id: str
    content_summary: str
    protected_entities: List[str] = field(default_factory=list)
