import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch


class CODataset(data.Dataset):
    """
    dataloader
    """

    def __init__(self, image_root, gt_root, edge_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)

        # self.images =self.read_files(image_root)
        # self.gts=self.read_files(gt_root)

        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.png')]
        #
        self.images = sorted(self.images)
        # print('images path :',self.images)
        self.gts = sorted(self.gts)
        # print('gts path :',self.gts)
        self.edges = sorted(self.edges)
        # print('depths path :',self.depths)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == 'True':
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        edge = self.binary_loader(self.edges[index])

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)

        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
            edge = self.gt_transform(edge)
        # image = torch.cat((image, depth), dim=0)
        return image, gt, edge

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        edges = []
        for img_path, gt_path, edge_path in zip(self.images, self.gts, self.edges):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            edge = Image.open(edge_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
                edges.append(edge_path)
        self.images = images
        self.gts = gts
        self.edges = edges

    # def read_files(self,root_file):
    #     sec_fold = os.listdir(root_file)
    #     sec_fold.sort()
    #     # print(sec_fold)
    #     image_path = []
    #     for sec in sec_fold:
    #         #print('root pth + sec',root_file + sec)
    #         #tempo_path = [root_file + sec + '/' + f for f in os.listdir(root_file + sec) if f.endswith('.jpg') or f.endswith('.png')]
    #         tempo_path = [root_file  + '/' + f for f in os.listdir(root_file ) if f.endswith('.jpg') or f.endswith('.png')]
    #         image_path = image_path + tempo_path
    #         #print(image_path, len(image_path))
    #     print(len(image_path))
    #     return image_path

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True,
               augmentation=False):
    dataset = CODataset(image_root, gt_root, edge_root, trainsize, augmentation)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.depth_transform = transforms.Compose([
        #     transforms.Resize((self.testsize, self.testsize)),
        #     transforms.ToTensor()
        # ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        # depth = self.binary_loader(self.depths[self.index])
        # 将深度图转换为Tensor并调整维度
        # depth = self.depth_transform(depth).unsqueeze(0)  # 转为Tensor并增加batch维度
        # 拼接RGB图像和深度图，形成四通道图像
        # image = torch.cat((image, depth), dim=1)  # 沿着通道维度拼接
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        # self.index += 1
        # self.index = self.index % self.size
        # return image, gt, name
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


class My_test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        # self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        # self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.depth_transform = transforms.Compose([
        #     transforms.Resize((self.testsize, self.testsize)),
        #     transforms.ToTensor()
        # ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        # depth = self.binary_loader(self.depths[self.index])
        name = self.images[self.index].split('/')[-1]
        # 将深度图转换为Tensor并调整维度
        # depth = self.depth_transform(depth).unsqueeze(0)  # 转为Tensor并增加batch维度
        # 拼接RGB图像和深度图，形成四通道图像
        # image = torch.cat((image, depth), dim=1)  # 沿着通道维度拼接
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
