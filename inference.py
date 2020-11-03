def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch inference script")
    parser.add_argument("--name", default=None, type=str, help="Name script")
    parser.add_argument("--checkpoint", default=None, type=str, help="Checkpoint file")
    parser.add_argument("--file", default=None, type=str, help="Path to input file")
    parser.add_argument("--output", default=None, type=str, help="Path to output file")
    args = parser.parse_args()
    main(args)
