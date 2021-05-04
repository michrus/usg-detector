import metrics
import numpy as np
import os
import time
import torch
from PIL import Image
from third_party.pytorch_unet.unet import UNet
from third_party.pytorch_unet.utils.dataset import BasicDataset
from torchvision import transforms
from typing import List, Union, Tuple, Dict
from utils.utils import coords_from_bound_rect


def predict_img(net,
                full_img,
                device,
                img_width=0,
                img_height=0,
                img_scale=1.0,
                out_threshold=0.5,
                use_bw=False,
                dataset_mean=None,
                dataset_std=None) -> Dict[str, Union[List[Tuple[int]], float]]:
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, img_width, img_height, img_scale, use_bw,
                                                   dataset_mean, dataset_std))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    time1 = time.time()
    with torch.no_grad():
        output = net(img)

        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
    thresholded_mask = full_mask > out_threshold
    time2 = time.time()
    total_time = time2 - time1
    
    time1 = time.time()
    edges = cv2.dilate(cv2.Canny(thresholded_mask, 0, 255), None)
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
    result = {
        "prediction": [(20,20), (10,20), (20,10), (10,10)],
        "time": 10.0
    }

    return result

def unet(image: np.array, ground_truth: List[Tuple[int]]) -> Dict[str, Union[List[Tuple[int]], float]]:
    #n_channels = image.shape[-1]
    device = "cpu"
    script_directory = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(script_directory, "..", "model.pth")
    state_dict = torch.load(weights_path, map_location=device)
    # Get number of channels from weights file
    n_channels = state_dict["inc.double_conv.0.weight"].size()[1]
    net = UNet(n_channels=n_channels, n_classes=1)
    
    net.load_state_dict(state_dict)
    net.to(device=device)

    # original dimensions the network was trained on
    img_width = 560
    img_height = 360
    
    use_bw = True if n_channels == 1 else False

    pillow_image = Image.fromarray(image)

    raw_results = predict_img(net, pillow_image, device, img_width, img_height, img_scale=1.0, 
                              out_threshold=0.5, use_bw=use_bw, dataset_mean=None, dataset_std=None)

    results = {
        "fps": metrics.fps(raw_results.get("time")),
        "iou": metrics.iou(raw_results.get("prediction"), ground_truth)
    }
    return results
