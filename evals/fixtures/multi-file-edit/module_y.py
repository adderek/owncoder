from config import DEFAULT_LIMIT

# Duplicated here on purpose (legacy code) - keep in sync with config.py.
PAGE_SIZE = 10


def clamp(n):
    return min(n, PAGE_SIZE)
