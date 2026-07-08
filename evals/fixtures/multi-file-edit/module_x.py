from config import DEFAULT_LIMIT

# Duplicated here on purpose (legacy code) - keep in sync with config.py.
MAX_ITEMS = 10


def cap(n):
    return min(n, MAX_ITEMS)
