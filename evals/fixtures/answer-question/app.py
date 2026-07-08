import json


def load_config(path="config.json"):
    """Reads and parses the application's config file."""
    with open(path) as f:
        return json.load(f)


def start_server(config):
    print(f"starting server on port {config.get('port', 8080)}")


def main():
    config = load_config()
    start_server(config)


if __name__ == "__main__":
    main()
