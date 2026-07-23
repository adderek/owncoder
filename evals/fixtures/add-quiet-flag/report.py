import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="world")
    args = parser.parse_args()
    print(f"generating report for {args.name}")
    print("done")


if __name__ == "__main__":
    main()
