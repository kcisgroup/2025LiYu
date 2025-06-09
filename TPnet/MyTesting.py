import torch
import torch.nn.functional as F
import numpy as np
import os, argparse

from matplotlib import pyplot as plt
import cv2
import time

from lib.pvt import TPnet, Hitnet
from utils.dataloader import My_test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=704, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='./Dataset/res/0.0274/TPnet.pth')
opt = parser.parse_args()

start_time = time.time()  # 记录开始时间
# for _data_name in ['CAMO', 'COD10K', 'CHAMELEON',NC4K]:
for _data_name in ['COD']:
# for _data_name in [d for d in os.listdir('D:/mvtec_anomaly_detection/') if os.path.isdir(os.path.join('D:/mvtec_anomaly_detection/', d))]:
    data_path = './Dataset/TestDataset/{}/'.format(_data_name)
    # data_path = 'D:/mvtec_anomaly_detection/{}/'.format(_data_name)
    save_path = './Dataset/all/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)

    model = TPnet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    # torch.save(model.state_dict(), save_path + 'TPnet.pth')
    image_root = '{}/Image/'.format(data_path)
    # image_root = '{}/test/'.format(data_path)
    # gt_root = '{}/ground_truth/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    edge_root = '{}/Edge/'.format(data_path)
    print('root', image_root, gt_root)
    test_loader = My_test_dataset(image_root, gt_root, opt.testsize)
    print('****', test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        print('***name', name)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        P1, P2, edge = model(image)
        res = F.upsample(P1[-1]+P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+name, res*255)

end_time = time.time()  # 记录结束时间
elapsed_time = end_time - start_time  # 计算耗时
print(f"程序运行耗时: {elapsed_time:.6f}秒")
