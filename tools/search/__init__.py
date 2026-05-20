from agent.tools.search.main import (
    setup as _setup_main,
    search_code,
    search_archive,
)
from agent.tools.search.grep import (
    setup as _setup_grep,
    grep_code,
)


def setup(config, data_provider) -> None:
    _setup_main(config, data_provider)
    _setup_grep(config)


__all__ = [
    "setup",
    "search_code",
    "search_archive",
    "grep_code",
]
