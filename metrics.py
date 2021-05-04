from typing import List, Union, Tuple


def iou(predicted_box: List[Tuple[int]], ground_truth_box: List[Tuple[int]]) -> float:
    x_a = max(predicted_box[0][0], ground_truth_box[0][0])
    y_a = max(predicted_box[0][1], ground_truth_box[0][1])
    x_b = min(predicted_box[-1][0], ground_truth_box[-1][0])
    y_b = min(predicted_box[-1][1], ground_truth_box[-1][1])
    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    predicted_box_area = (predicted_box[-1][0] - predicted_box[0][0] + 1) * (predicted_box[-1][1] - predicted_box[0][1] + 1)
    ground_truth_box_area = (ground_truth_box[-1][0] - ground_truth_box[0][0] + 1) * (ground_truth_box[-1][1] - ground_truth_box[0][1] + 1)
    iou = intersection_area / float(predicted_box_area + ground_truth_box_area - intersection_area)
    return iou

def fps(time: float, batch_size: int = 1) -> float:
    return batch_size/time
