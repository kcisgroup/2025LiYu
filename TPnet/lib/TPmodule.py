import torch
from matplotlib import pyplot as plt
from skimage import color
from scipy.spatial.distance import cdist
import cv2
import numpy as np
from skimage.filters import gabor
from torchvision import transforms


# 选择前10的特征通道
def FreqStatRank(ft, image):
    _, dim, h, w = ft.shape
    ft = ft.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    I1 = cv2.resize(image_rgb, (w, h))
    I2 = color.rgb2hsv(I1)
    V = I2[:, :, 2]  # 获取 Value 组件
    # 使用Gabor滤波器的各个方向
    ang = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    # 应用Gabor滤波器并计算幅值响应
    gabor_responses = []
    for angle in ang:
        filt_real, filt_imag = gabor(V, frequency=2.0, theta=np.deg2rad(angle))
        gabor_responses.append(np.sqrt(filt_real ** 2 + filt_imag ** 2))
    gabortmp = np.array(gabor_responses)
    E = np.sum(gabortmp, axis=(1, 2))
    rft = cv2.resize(ft, (9, 9))
    sft = np.sum(rft, axis=0)
    sft = np.reshape(sft, (9, dim))
    sft = sft.T
    dist = cdist(E.reshape(1, -1), sft, metric='euclidean')
    index = np.argsort(dist.flatten())
    m = 10
    mo = index[:m]
    ms = np.zeros((h, w))
    # 全局拓扑特征图
    for k in range(0, m):
        ms = ms + ft[:, :, mo[k]]
    ms = torch.tensor(ms).unsqueeze(0)
    return ms
