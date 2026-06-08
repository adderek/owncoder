import re
import json
from typing import List, Dict, Any, Optional
from agent.memory.h_compaction_types import ProtectedEntity

class SemanticIntegrityError(Exception):
    """Raised when the summary diverges significantly from the original intent."""
    pass

class KnowledgeDoSException(Exception):
    """Raised when input data is detected as maliciously empty or useless."""
    pass

class EntityProtector:
    """Ensures technical terms are identified and protected during summarization."""
    
    def __init__(self, patterns: Dict[str, str] = None):
        # Default patterns for common technical entities
        self.patterns = patterns or {
            # File paths with known extensions — specific enough not to over-match prose
            "file": (
                r"\b[\w./\-]+"
                r"\.(?:py|js|ts|jsx|tsx|yaml|yml|json|toml|md|sh|txt|cfg|ini|sql|rs|go|c|cpp|h)\b"
            ),
            # Hex addresses / magic values
            "hex": r"\b0x[0-9a-fA-F]+\b",
            # Explicit error codes / tags
            "error_code": r"\bERROR[:\s]+\w+",
        }
        self.protected_entities: List[ProtectedEntity] = []

    def scan_text(self, text: str) -> List[ProtectedEntity]:
        found = []
        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                term = match.group(0)
                found.append(ProtectedEntity(term=term, entity_type=entity_type))
        return found

    def protect_text(self, text: str, entities: List[ProtectedEntity]) -> str:
        """Wraps entities in markers to prevent LLM from altering them."""
        protected_text = text
        # Sort by length descending to prevent partial replacement issues
        sorted_entities = sorted(entities, key=lambda x: len(x.term), reverse=True)
        
        # Use a set to avoid double-protecting the same term
        seen_terms = set()
        
        for entity in sorted_entities:
            if entity.term in seen_terms:
                continue
            
            # Check if term is already wrapped
            if f"[[{entity.term}]]" in protected_text:
                continue

            # Replace all occurrences of this term with wrapped version
            # We use a regex with word boundaries to avoid partial matches if possible, 
            # but for simplicity in this implementation, we use direct replacement.
            protected_text = protected_text.replace(entity.term, f"[[{entity.term}]]")
            seen_terms.add(entity.term)
            
        return protected_text

    def unprotect_text(self, text: str) -> str:
        """Removes the [[ ]] markers."""
        return re.sub(r"\[\[(.*?)\]\]", r"\1", text)


class SmartCompactor:
    """Summarizes memory segments while protecting technical entities."""

    def __init__(self, entity_protector: EntityProtector, similarity_threshold: float = 0.3):
        self.entity_protector = entity_protector
        self.similarity_threshold = similarity_threshold

    def summarize_segment(self, segment_text: str) -> str:
        stripped = segment_text.strip()
        if not stripped or len(stripped) < 3:
            raise KnowledgeDoSException(f"Input too short to summarize: {repr(segment_text)}")

        entities = self.entity_protector.scan_text(segment_text)
        terms = [e.term for e in entities]

        if terms:
            term_list = ", ".join(sorted(set(terms)))
            summary = f"Summary of segment containing: {term_list}"
        else:
            summary = f"Summary of segment with no specific entities."

        similarity = self._calculate_semantic_similarity(segment_text, summary)
        if similarity < self.similarity_threshold:
            raise SemanticIntegrityError(
                f"Summary diverged from original (similarity={similarity:.2f} < {self.similarity_threshold})"
            )

        return summary

    def _calculate_semantic_similarity(self, text_a: str, text_b: str) -> float:
        words_a = set(re.findall(r"\w+", text_a.lower()))
        words_b = set(re.findall(r"\w+", text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
