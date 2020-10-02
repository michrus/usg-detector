import argparse
import os
# import numpy as np
os.environ["PYTORCH_JIT"] = "0"
import torch
# import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
# import matplotlib.patches as mpatches
import operator
from skimage.measure import label, regionprops

from third_party.pytorch_unet.unet import UNet
from third_party.pytorch_unet.utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

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

    return full_mask > out_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filename of input images', required=True)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    args = parser.parse_args()
    in_file = args.input

    net = UNet(n_channels=3, n_classes=1)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    img = Image.open(in_file)

    mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=args.scale,
                        out_threshold=args.mask_threshold,
                        device=device)

    # image = np.array(img)
    # label image regions
    label_image = label(mask)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(image)
    
    regps = regionprops(label_image)
    max_region = max(regps, key=operator.attrgetter("area"))
    miny, minx, maxy, maxx = max_region.bbox
    # rect = mpatches.Rectangle((minx, miny), maxx - minx, maxy - miny,
                            # fill=False, edgecolor='red', linewidth=1)
    # ax.add_patch(rect)

    print(f"{minx},{miny},{maxx},{maxy}")

    # ax.set_axis_off()
    # plt.tight_layout()
    # plt.show()