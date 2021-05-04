import cv2
import metrics
import numpy as np
import time
from typing import List, Union, Tuple, Dict
from utils.utils import coords_from_bound_rect


def results_mean(image: np.array, ground_truth: List[Tuple[int]]):
    raw_results = segmentation(image, ground_truth, "mean")
    results = {
        "fps": metrics.fps(raw_results.get("time")),
        "iou": metrics.iou(raw_results.get("prediction"), ground_truth)
    }
    return results

def results_otsu(image: np.array, ground_truth: List[Tuple[int]]):
    raw_results = segmentation(image, "otsu")
    results = {
        "fps": metrics.fps(raw_results.get("time")),
        "iou": metrics.iou(raw_results.get("prediction"), ground_truth)
    }
    return results

def segmentation(image: np.array, thresholding_type: str) -> Dict[str, Union[List[Tuple[int]], float]]:
    time1 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    time2 = time.time()
    total_time = time2 - time1
 
    if thresholding_type == "mean":
        time1 = time.time()
        _,thresh = cv2.threshold(gray,gray.mean(),255,cv2.THRESH_BINARY)
        time2 = time.time()
    elif thresholding_type == "otsu":
        time1 = time.time()
        _,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        time2 = time.time()
    else:
        ValueError("Only mean and otsu implemented as thresholding type for classic segmentation!")
    total_time = total_time + (time2 - time1)

    time1 = time.time()
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    time2 = time.time()
    total_time = total_time + (time2 - time1)

    time1 = time.time()
    contour = sorted(cv2.findContours(edges, cv2.RETR_LIST, 
                                        cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    time2 = time.time()
    total_time = total_time + (time2 - time1)

    time1 = time.time()
    x, y, w, h = cv2.boundingRect(contour)
    box = coords_from_bound_rect(x, y, w, h)
    time2 = time.time()
    total_time = total_time + (time2 - time1)

    result = {
        "prediction": box,
        "time": total_time
    }
    return result
