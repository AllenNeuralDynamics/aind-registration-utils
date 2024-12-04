"""
Register individual to template with points
"""

import argparse

from aind_registration_utils.recipes import (
    individual_to_template_with_points_files,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find the transformation from an individual image to a "
            "template image and apply it to a set of points."
        )
    )
    parser.add_argument(
        "individual", help="path to the image of an individual"
    )
    parser.add_argument("mask", help="path to the brain mask of an individual")
    parser.add_argument("template", help="path to the image of the template")
    parser.add_argument(
        "targets", help="path to the targets in the template space"
    )
    parser.add_argument("output", nargs="?", help="output directory")
    parser.add_argument("-m", "--mouse", default=None, help="mouse ID")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="force overwrite",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    individual_to_template_with_points_files(
        args.individual,
        args.mask,
        args.template,
        args.targets,
        args.output,
        mouse_id=args.mouse,
    )


if __name__ == "__main__":
    main()
