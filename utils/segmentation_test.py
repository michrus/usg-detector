import argparse
from os.path import join
import cv2
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from PIL import Image
from typing import List, Union, Optional

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
        if args.v:
            log.setLevel(logging.DEBUG)
        segmentation(args.i, args.save_dir)
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
    parser.add_argument("-i",
                        help=("Input image.\n"),
                        action="store", type=str, required=True)
    parser.add_argument("-v",
                        help="Print more verbose logs output.",
                        action="store_true", required=False)
    parser.add_argument("--save_dir",
                        help=("Directory for visualization images.\n"),
                        action="store", type=str, required=False)
    args = parser.parse_args()
    if not os.path.exists(args.i):
        parser.error("Input image does not exist!")
        sys.exit(1)
    if os.path.isdir(args.i):
        parser.error("Path given as input image is directory!")
        sys.exit(1)
    if args.save_dir and not os.path.exists(args.save_dir):
        parser.error("Save directory does not exist!")
        sys.exit(1)

    return args

def segmentation(image_path: str, save_dir: Optional[str]) -> None:
    w=15
    h=15
    columns = 3
    rows = 2
    fig=plt.figure(figsize=(8, 8))
    plt.axis('off')
    
    log.debug(f"Loading image.")
    image = np.asarray(Image.open(image_path), dtype=np.uint8)
    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)  

    log.debug(f"Converting to grayscale.")
    time1 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    time2 = time.time()
    total_time = time2 - time1
    fig.add_subplot(rows, columns, 2)
    plt.imshow(gray, cmap='gray', vmin=0, vmax=255)

    log.debug(f"Applying thresholding.")
    #edges = cv2.dilate(cv2.Canny(mask, 254, 255), None)
    # _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    #         cv2.THRESH_BINARY_INV,19,2)
    time1 = time.time()
    _,thresh = cv2.threshold(gray,gray.mean(),255,cv2.THRESH_BINARY)
    time2 = time.time()
    total_time = total_time + (time2 - time1)
    fig.add_subplot(rows, columns, 3)
    plt.imshow(thresh, cmap='gray', vmin=0, vmax=255)

    log.debug(f"Detecting edges.")
    time1 = time.time()
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    # edges = cv2.Canny(thresh, 0, 255)
    time2 = time.time()
    total_time = total_time + (time2 - time1)
    fig.add_subplot(rows, columns, 4)
    plt.imshow(edges, cmap='gray', vmin=0, vmax=255)

    log.debug(f"Finding largest contour.")
    time1 = time.time()
    contour = sorted(cv2.findContours(edges, cv2.RETR_LIST, 
                                        cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    time2 = time.time()
    total_time = total_time + (time2 - time1)

    log.debug(f"Getting bounding box.")
    time1 = time.time()
    x, y, w, h = cv2.boundingRect(contour)
    box = coords_from_bound_rect(x, y, w, h)
    time2 = time.time()
    total_time = total_time + (time2 - time1)
    cv2.rectangle(image,box[0],box[-1],(0,191,255),1)
    fig.add_subplot(rows, columns, 5)
    plt.imshow(image)

    log.info(f"Processing time {total_time*1000} ms")

    if save_dir:
        file_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, file_name)
        log.debug(f"Saving bounding box image to: {save_path}")
        cv2.imwrite(save_path, image)

    plt.show()

def coords_from_bound_rect(x: float, y: float, w: float, h: float) -> List[float]:
    x1 = x
    y1 = y
    x2 = x+w
    y2 = y+h
    return [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]

if __name__ == "__main__":
    sys.exit(main())
