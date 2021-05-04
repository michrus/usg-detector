import argparse
import csv
import glob
import logging
import numpy as np
import os
import segmentation.classic
import sys
from os.path import join
from PIL import Image
from statistics import mean
from typing import List, Union, Tuple, Dict


logging.basicConfig(format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

TASK_METHODS_MAPPING: Dict[str, List[str]] = {
    "segmentation": {
        "classic_mean_threshold": segmentation.classic.results_mean,
        "classic_otsu_threshold": segmentation.classic.results_otsu
    }
}
AVAILABLE_METHODS = [method for methods in TASK_METHODS_MAPPING.values() for method in methods]

def main() -> int:
    """
    Main entry to execute script.

    :param argv: cli params
    :return: int
    """
    return_code = 0
    
    # try:
    args = parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)
    image_tasks = [args.task] if args.task is not list else args.task
    if "all" in image_tasks:
        image_tasks = list(TASK_METHODS_MAPPING.keys())
    methods = [args.method] if args.method is not list else args.method
    if "all" in methods:
        methods = AVAILABLE_METHODS
    labels = get_labels(args.labels)
    per_image_results = get_per_image_results(args.images, labels, image_tasks, methods)
    # import pprint
    # pprint.pprint(per_image_results)
    final_results = get_final_results(per_image_results)
    print_results(final_results, args.images)
    # except Exception as e:
    #     log.error(f"Fatal error: {e}")
    #     return_code = 255
    log.info("Script finished working.")
    return return_code

def parse_args() -> argparse.ArgumentParser:
    """
    Parse script arguments.

    :return: argparse.ArgumentParser
    """
    log.info(f"Parsing script arguments.")
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--images", "-i",
                        help=("Input directory with images.\n"),
                        action="store", type=str, required=True)
    parser.add_argument("--labels", "-l",
                        help=("Path to labels file.\n"),
                        action="store", type=str, required=True)
    parser.add_argument("--task", "-t",
                        help=("Image tasks to perform.\n"),
                        nargs="?", choices=["all", "segmentation"], 
                        default="all", type=str, required=False)
    parser.add_argument("--method", "-m",
                        help=("Methods for image tasks.\n"),
                        nargs="?", choices=["all", *AVAILABLE_METHODS], 
                        default="all", type=str, required=False)
    parser.add_argument("--weights", "-w",
                        help=("Path to weights directory for machine learning methods.\n"),
                        action="store", type=str, required=False)    
    parser.add_argument("--verbose", "-v",
                        help="Print more verbose logs output.",
                        action="store_true", required=False)
    parser.add_argument("--visualize",
                        help="Save images with bounding boxes.",
                        action="store_true", required=False)
    parser.add_argument("--vis_dir",
                        help=("Directory for visualization images.\n"),
                        action="store", type=str, required=False)
    args = parser.parse_args()
    if not os.path.exists(args.images):
        parser.error("Input directory does not exist!")
        sys.exit(1)
    if not os.path.exists(args.labels):
        parser.error("Labels file does not exist!")
        sys.exit(1)
    if args.vis_dir and not os.path.exists(args.vis_dir):
        parser.error("Input directory does not exist!")
        sys.exit(1)

    return args

def get_labels(labels_path: str) -> List[Dict[str, str]]:
    log.info(f"Reading labels file: {labels_path}")
    labels: List[Dict[str, str]] = []
    with open(labels_path) as csv_file:
        reader = csv.DictReader(csv_file)
        labels = [row for row in reader]
    return labels

def get_per_image_results(images_dir: str, labels: List[Dict[str, str]], image_tasks: List[str], methods: List[str]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    log.info("Getting results for each image")
    results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for row in labels:
        image_name = row.get("file_name")
        image_path = os.path.join(images_dir, image_name)
        ground_truth = get_ground_truth(row)
        log.debug(f"Reading image: {image_path}")
        image = np.asarray(Image.open(image_path))
        image_results = get_image_results(image, ground_truth, image_tasks, methods)
        results[image_name] = image_results
    return results

def get_ground_truth(label_row: Dict[str,str]) -> List[Tuple[int]]:
    file_name = label_row["file_name"]
    log.debug(f"Parsing ground truth for {file_name}")
    coordinates: List[Tuple(int)] = []
    for i in range(0,4):
        x = int(label_row.get(f"x{i}"))
        y = int(label_row.get(f"y{i}"))
        coordinates.append((x,y))
    log.debug(f"Ground truth is: {coordinates}")
    return coordinates

def get_image_results(image: np.array, ground_truth: List[Tuple[int]], image_tasks: List[str], methods: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task in image_tasks:
        results[task] = get_results_for_task(image, ground_truth, task, methods)
    return results

def get_results_for_task(image: np.array, ground_truth: List[Tuple[int]], task: str, used_methods: List[str]) -> Dict[str, Dict[str, float]]:
    log.debug(f"Getting results for task: {task}")
    results: Dict[str, Dict[str, float]] = {}
    for method in TASK_METHODS_MAPPING.get(task):
        if method in used_methods:
            results[method] = get_results_for_method(image, ground_truth, task, method)
        else:
            continue
    return results

def get_results_for_method(image: np.array, ground_truth: List[Tuple[int]], task: str, method: str) -> Dict[str, float]:
    log.debug(f"Getting results for method: {method}")
    results = (TASK_METHODS_MAPPING.get(task).get(method))(image, ground_truth)
    log.debug(f"Results: {results}")
    return results

def get_final_results(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    aggregate_results = {}
    for image in results.keys():
        for task in results[image].keys():
            if not aggregate_results.get(task):
                aggregate_results[task] = {}
            for method in results[image][task].keys():
                if not aggregate_results[task].get(method):
                    aggregate_results[task][method] = {}
                for metric in results[image][task][method].keys():
                    if not aggregate_results[task][method].get(metric):
                        aggregate_results[task][method][metric] = []
                    aggregate_results[task][method][metric].append(results[image][task][method][metric])
    # calculate averages
    final_results = {}
    for task in aggregate_results.keys():
        if not final_results.get(task):
            final_results[task] = {}
        for method in aggregate_results[task].keys():
            if not final_results[task].get(method):
                final_results[task][method] = {}
            for metric in aggregate_results[task][method].keys():
                metric_average_key = f"{metric}_avg"
                final_results[task][method][metric_average_key] = mean(aggregate_results[task][method][metric])
    return final_results

def print_results(final_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], images_dir: str) -> None:
    log.info("**************** FINAL RESULTS ****************")
    log.info(f"Results achieved on images in directory: {images_dir}")
    for task in final_results.keys():
        log.info(f"==== TASK: {task}")
        for method in final_results[task].keys():
            log.info(f"======== METHOD: {method}")
            for metric in final_results[task][method].keys():
                log.info(f"============ METRIC: {metric} = {final_results[task][method][metric]}")

if __name__ == "__main__":
    sys.exit(main())
