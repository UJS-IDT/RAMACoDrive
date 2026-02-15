import argparse


def recieve_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--debug",  # you can use it to change world settings
        default="train",
        help="Enable debug mode and change the conifg data to load on a small map",
    )
    parser.add_argument(
        "-dr",
        "--draw",
        action="append",  # you can use -dr multiple times
        help="Enable draw mode and draw the plot of the obstacle sensor",
    )
    # townnam
    parser.add_argument(
        "-t",
        "--town",
        type=str,
        help="Choose the town to load",
    )
    parser.add_argument(
        "-hy",
        "--hypes",
        type=str,
        help="hypes of method to load",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        default='',
        help="hypes of method to load",
    )
    parser.add_argument(
        '--show_sequence',
        action='store_true',
        help='whether to show video visualization result.',
    )
    parser.add_argument(
        "-ep",
        "--epoches",
        type=int,
        help="rest_epoches",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        choices=["file", "console", "all", "none", ""],
        help="Enable log",
    )
    args = parser.parse_args()
    return args
