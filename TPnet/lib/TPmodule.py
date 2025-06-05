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
    # 第一个参数ft为深度神经网络提取的特征，第二个参数为图像本身（深度图像还是图像本身？）
    _, dim, h, w = ft.shape
    ft = ft.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 从 [1, 3, 352, 352] 变为 [352, 352, 3]，以适应opencv的处理
    # 显示图像
    # plt.imshow(image)
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
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
    # 将结果转换为三维数组
    gabortmp = np.array(gabor_responses)
    # 在第一维和第二维进行求和
    E = np.sum(gabortmp, axis=(1, 2))
    # 将卷积特征调整为9x9的尺寸
    rft = cv2.resize(ft, (9, 9))
    # 对每列进行求和，得到9xdim的矩阵，再进行转置
    sft = np.sum(rft, axis=0)
    sft = np.reshape(sft, (9, dim))
    sft = sft.T
    # 计算纹理特征与卷积特征之间的欧几里得距离
    dist = cdist(E.reshape(1, -1), sft, metric='euclidean')
    # 对距离进行升序排序，返回升序排列的索引
    index = np.argsort(dist.flatten())
    # 选择最小距离前10的通道索引，即前10个最匹配的特征通道
    m = 10
    mo = index[:m]
    ms = np.zeros((h, w))
    # 全局拓扑特征图
    for k in range(0, m):
        ms = ms + ft[:, :, mo[k]]
    ms = torch.tensor(ms).unsqueeze(0)
    return ms
    # print(ms.shape)
    # ms = cv2.resize(ms, (448, 448))
    # # 创建一个 1x2 的子图布局
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 行 2 列
    # # 在第一个子图中显示第一张图片
    # axes[0].imshow(image)
    # axes[0].axis('off')  # 关闭坐标轴
    # axes[0].set_title("Image 1")  # 添加标题
    #
    # # 在第二个子图中显示第二张图片
    # axes[1].imshow(ms)
    # axes[1].axis('off')  # 关闭坐标轴
    # axes[1].set_title("Image 2")  # 添加标题
    #
    # # 调整布局，使子图之间不会重叠
    # plt.tight_layout()
    #
    # # 显示图片
    # plt.show()
