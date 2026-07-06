import argparse


def main():
    parser = argparse.ArgumentParser(description="Greet someone.")
    parser.add_argument("--name", default="world", help="name to greet")
    # TODO: add a --verbose flag that prints "verbose on" when set.
    args = parser.parse_args()
    print(f"hello {args.name}")


if __name__ == "__main__":
    main()
