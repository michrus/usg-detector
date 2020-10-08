import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import matplotlib.patches as mpatches
import operator
from skimage.measure import label, regionprops

from third_party.pytorch_unet.unet import UNet
from third_party.pytorch_unet.utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                img_width=0,
                img_height=0,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, img_width, img_height))

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
                        help='filename of input image', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT',
                        help='filename of output image (should be .png)', required=True)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('-r', '--resize', dest='resize_string', type=str,
                        help='Size images should be resized to, in format: NxM. Example: 24x24')

    args = parser.parse_args()
    in_file = args.input

    net = UNet(n_channels=3, n_classes=1)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    img = Image.open(in_file)#.convert("RGB")

    if args.resize_string:
        resize = list(map(int, args.resize_string.split("x")))
        img_width = resize[0]
        img_height = resize[1]
    else:
        img_width = 0
        img_height = 0

    mask = predict_img(net=net,
                        full_img=img,
                        img_width=img_width,
                        img_height=img_height,
                        out_threshold=args.mask_threshold,
                        device=device)

    image = np.array(img)
    # label image regions
    label_image = label(mask)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    
    regps = regionprops(label_image)
    max_region = max(regps, key=operator.attrgetter("area"))
    miny, minx, maxy, maxx = max_region.bbox
    rect = mpatches.Rectangle((minx, miny), maxx - minx, maxy - miny,
                            fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(args.output)