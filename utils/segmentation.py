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
        csv_data = get_labels_csv(args.masks_dir, args.visualize, args.vis_dir)
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
                        help="Print more verbose logs output.",
                        action="store_true", required=False)
    parser.add_argument("--visualize",
                        help="Save images with bounding boxes.",
                        action="store_true", required=False)
    parser.add_argument("--vis_dir",
                        help=("Directory for visualization images.\n"),
                        action="store", type=str, required=False)
    args = parser.parse_args()
    if not os.path.exists(args.masks_dir):
        parser.error("Input directory does not exist!")
        sys.exit(1)
    if args.vis_dir and not os.path.exists(args.vis_dir):
        parser.error("Input directory does not exist!")
        sys.exit(1)

    return args

def get_labels_csv(input_dir_path: str, visualize: bool = False, vis_dir: str = "") -> List[List[str]]:
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
        result_csv.append(get_label(file_path, visualize, vis_dir))

    return result_csv

def get_label(file_path: str, visualize: bool = False, vis_dir: str = "") -> List[str]:
    """
    Finds corner coordinates that make single row of csv data.

    :return: list of string
    """
    image = np.asarray(Image.open(file_path), dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #edges = cv2.dilate(cv2.Canny(mask, 254, 255), None)
    # _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,thresh = cv2.threshold(gray,gray.mean(),255,cv2.THRESH_BINARY)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    contour = sorted(cv2.findContours(edges, cv2.RETR_LIST, 
                                      cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    x, y, w, h = cv2.boundingRect(contour)
    box = coords_from_bound_rect(x, y, w, h)
    # rect = cv2.minAreaRect(contour)
    # box = cv2.boxPoints(rect)
    box_string = [str(coord) for pair in box for coord in pair]
    file_name = os.path.basename(file_path)
    result = [file_name, *box_string]
    log.debug(f"{result}")
    if (visualize):
        box_image(image, box, vis_dir, file_name)
    return result

def coords_from_bound_rect(x: float, y: float, w: float, h: float) -> List[float]:
    x1 = x
    y1 = y
    x2 = x+w
    y2 = y+h
    return [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]

def box_image(image: np.ndarray, box: List[List[float]], 
                     box_images_dir: str, file_name: str) -> None:
    #image_rgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    cv2.rectangle(image,box[0],box[-1],(0,191,255),1)
    save_path = os.path.join(box_images_dir, file_name)
    log.debug(f"Saving bounding box image to: {save_path}")
    cv2.imwrite(save_path, image)

def write_csv(output_dir: str, csv_data: List[List[str]]) -> None:
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
