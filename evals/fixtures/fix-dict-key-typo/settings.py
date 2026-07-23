CONFIG = {
    "timeout_seconds": 30,
    "retries": 3,
}


def get_timeout(config):
    return config["timeout_secodns"]  # bug: typo'd key
