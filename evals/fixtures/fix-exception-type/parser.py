def parse_int(text):
    try:
        return int(text)
    except TypeError:  # bug: int(text) on bad text raises ValueError, not TypeError
        return None
