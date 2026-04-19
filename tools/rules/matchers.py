import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PathMatcher:
    """Gitignore-style path matcher using the pathspec library."""

    def __init__(self, patterns: list[str] | None = None):
        self._patterns = patterns or []
        self._spec = None
        if self._patterns:
            try:
                import pathspec

                self._spec = pathspec.PathSpec.from_lines(
                    "gitwildmatch", self._patterns
                )
            except ImportError:
                logger.warning("pathspec not installed — path rule matching disabled")

    def matches(self, path: str) -> bool:
        if self._spec is None:
            return False
        return self._spec.match_file(path)

    @property
    def empty(self) -> bool:
        return not self._patterns


class ReadonlyMatcher:
    """Path matcher that carries optional reason annotations."""

    def __init__(
        self, patterns: list[str] | None = None, reasons: dict[str, str] | None = None
    ):
        self._matcher = PathMatcher(patterns)
        self._reasons = reasons or {}
        self._patterns = patterns or []

    def matches(self, path: str) -> tuple[bool, str | None]:
        if not self._matcher.matches(path):
            return False, None
        # Find the most specific matching pattern's reason
        for pattern in reversed(self._patterns):
            if pattern in self._reasons:
                try:
                    import pathspec

                    spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern])
                    if spec.match_file(path):
                        return True, self._reasons[pattern]
                except Exception:
                    pass
        return True, None

    @property
    def empty(self) -> bool:
        return self._matcher.empty
