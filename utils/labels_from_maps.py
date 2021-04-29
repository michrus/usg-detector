import argparse
from os.path import join
import cv2
import glob
import logging
import numpy as np
import os
import sys
from PIL import Image
from typing import List, Union

logging.basicConfig(format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def main() -> int:
    """
    Main entry to execute script.

    :param argv: cli params
    :return: int
    """
    return_code = 0
    try:
        args = parse_args()
        if args.verbose:
            log.setLevel(logging.DEBUG)
        csv_data = get_labels_csv(args.masks_dir)
        write_csv(args.masks_dir, csv_data)

    except Exception as e:
        log.error(f"Fatal error: {e}")
        return_code = 255
    log.info("Script finished working.")
    return return_code

def parse_args() -> argparse.ArgumentParser:
    """
    Parse script arguments.

    :return: argparse.ArgumentParser
    """
    log.info(f"Parsing script arguments.")
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--masks_dir",
                        help=("Input directory masks.\n"),
                        action="store", type=str, required=True)
    parser.add_argument("--verbose",
                        help="Print more verbose logs output",
                        action="store_true", required=False)
    args = parser.parse_args()
    if not os.path.exists(args.masks_dir):
        parser.error("Input directory does not exist!")
        sys.exit(1)

    return args

def get_labels_csv(input_dir_path) -> List[List[str]]:
    """
    Creates CSV data with corner point coordinates labels for every mask image.

    :return: csv data in form of list of lists
    """
    log.info(f"Creating CSV.")
    header = ["file_name", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"]
    result_csv = [header]
    files_list = os.listdir(input_dir_path)
    for file_name in files_list:
        file_path = os.path.join(input_dir_path, file_name)
        result_csv.append(get_label(file_path))

    return result_csv

def get_label(file_path) -> List[str]:
    """
    Finds corner coordinates that make single row of csv data.

    :return: list of string
    """
    mask = np.asarray(Image.open(file_path))
    edges = cv2.dilate(cv2.Canny(mask, 0, 255), None)
    contour = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box_string = [str(coord) for pair in box for coord in pair]
    file_name = os.path.basename(file_path)
    result = [file_name, *box_string]
    log.debug(f"{result}")
    return result

def write_csv(output_dir, csv_data) -> None:
    """
    Writes data to csv file.

    :return: list of string
    """
    log.info(f"Writing CSV.")
    file_name="labels.csv"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w") as f:
        for row in csv_data:
            line = ",".join(row)
            f.write(f"{line}\n")

if __name__ == "__main__":
    sys.exit(main())
