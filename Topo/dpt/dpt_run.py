"""Compute depth maps for images in the input folder.
"""
import glob
import os

import numpy as np
import torch
import cv2
import argparse

from torchvision.transforms import Compose

from dpt.models import DPTDepthModule
from dpt.transforms import NormalizeImage, Resize, PrepareForNet
from dpt.util.io import read_image, write_depth


# 如果数据库里存在深度，则直接使用，如果不存在，则使用网络计算得出
def dpt_run(input_path, output_path="output_monodepth", optimize=True):
    """Run MonoDepthNN to compute depth maps.
    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_w = net_h = 384
    model = DPTDepthModule(path="weights/dpt_large.pt",
                           backbone="vitl16_384",
                           non_negative=True,
                           enable_attention_hooks=False, )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)
    # create output folder
    os.makedirs(output_path, exist_ok=True)
    # get input
    # img_names = glob.glob(os.path.join(input_path, "*png"))
    img_names = glob.glob(os.path.join(input_path, "*.jpg")) + \
                glob.glob(os.path.join(input_path, "*.jpeg")) + \
                glob.glob(os.path.join(input_path, "*.png"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")
    for ind, img_name in enumerate(img_names):
        if os.path.isdir(img_name):
            continue
        file_name, file_extension = os.path.splitext(img_name)
        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))
        # input

        img = read_image(img_name)

        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction,
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        write_depth(filename, prediction, file_extension, bits=2)

    print("finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", default="D:/PycharmProject/yolov8-master/data/test_data/images/AIGIs/", help="folder with input images")
    parser.add_argument("-o", "--output_path", default="D:/PycharmProject/yolov8-master/data/test_data/depths/AIGIs/", help="folder for output images")
    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.set_defaults(optimize=True)
    args = parser.parse_args()
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # compute depth maps
    dpt_run(
        args.input_path,
        args.output_path,
        args.optimize,
    )
