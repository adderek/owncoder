import unittest
from agent.memory.smart_compactor import EntityProtector, SmartCompactor, KnowledgeDoSException, SemanticIntegrityError

class TestSmartCompactor(unittest.TestCase):
    def setUp(self):
        self.protector = EntityProtector()
        # Use a low threshold for the general case
        self.compactor = SmartCompactor(self.protector, similarity_threshold=0.1)

    def test_entity_preservation(self):
        """Verify that technical terms are preserved in the summary process."""
        # Matches 'file' pattern (.py) and 'error' pattern (ERROR: ...)
        original_text = "ERROR: critical failure in main.py"
        summary = self.compactor.summarize_segment(original_text)
        self.assertTrue("main.py" in summary or "ERROR" in summary)

    def test_knowledge_dos_detection(self):
        """Verify that suspiciously short/empty input triggers KnowledgeDoSException."""
        with self.assertRaises(KnowledgeDoSException):
            self.compactor.summarize_segment(" ")
        
        with self.assertRaises(KnowledgeDoSException):
            self.compactor.summarize_segment("a")

    def test_semantic_integrity_error(self):
        """Verify that divergence triggers SemanticIntegrityError."""
        # A text with no entities will result in a summary with 0 similarity
        # if the threshold is high.
        original_text = "The quick brown fox jumps over the lazy dog."
        high_threshold_compactor = SmartCompactor(self.protector, similarity_threshold=0.9)
        with self.assertRaises(SemanticIntegrityError):
            high_threshold_compactor.summarize_segment(original_text)

    def test_successful_summarization(self):
        """Verify successful path with entities."""
        # Matches 'file' pattern
        original_text = "Check script.py for updates."
        summary = self.compactor.summarize_segment(original_text)
        self.assertTrue("script.py" in summary)

if __name__ == '__main__':
    unittest.main()