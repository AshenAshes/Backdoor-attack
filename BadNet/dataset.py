import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import time


class MyDataset(Dataset):
    def __init__(self, dataset, target, mask, portion, mode, device):
        self.dataset = self.addTrigger(dataset, target, portion, mode, mask)
        self.device = device

    def __getitem__(self, item):
        img = self.dataset[item][0]
        img = img[..., np.newaxis]
        img = torch.Tensor(img).permute(2, 0, 1)
        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label

    def __len__(self):
        return len(self.dataset)

    def addMask1(self, img, width, height):
        img[width - 3][height - 3] = 255
        img[width - 3][height - 2] = 255
        img[width - 2][height - 3] = 255
        img[width - 2][height - 2] = 255

    def addMask2(self, img, width, height):
        img[width - 10][height - 10] = 255
        img[width - 10][height - 10] = 255
        img[width - 10][height - 10] = 255
        img[width - 9][height - 9] = 255

    def addMask3(self, img, width, height):
        img[width - 3][height - 3] = 255

    def addMask4(self, img, width, height):
        img[width - 3][height - 3] = 255
        img[width - 3][height - 4] = 255
        img[width - 3][height - 5] = 255
        img[width - 3][height - 6] = 255
        img[width - 3][height - 7] = 255

    def addMask5(self, img, width, height):
        img[width - 3][height - 3] = 255
        img[width - 3][height - 4] = 255
        img[width - 3][height - 5] = 255
        img[width - 4][height - 5] = 255
        img[width - 5][height - 5] = 255
        img[width - 5][height - 4] = 255
        img[width - 5][height - 3] = 255
        img[width - 4][height - 3] = 255



    def addTrigger(self, dataset, target, portion, mode, mask):
        print("Generating " + mode + " Bad Imgs")
        perm = np.random.permutation(len(dataset))[0: int(len(dataset) * portion)]
        dataset_ = list()
        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = np.array(data[0])
            width = img.shape[0]
            height = img.shape[1]
            if i in perm:
                if mask == 1:  # 一个区域
                    self.addMask1(img, width, height)
                elif mask == 2:  # 区域在中间
                    self.addMask2(img, width, height)
                elif mask == 3:  # 区域放大
                    self.addMask3(img, width, height)
                elif mask == 4:  # 区域有意义
                    self.addMask4(img, width, height)
                elif mask == 5:  # 区域有意义
                    self.addMask4(img, width, height)
                dataset_.append((img, target))
                cnt += 1
            else:
                dataset_.append((img, data[1]))
        time.sleep(0.1)
        print("Injecting Over: " + str(cnt) + " Bad Imgs, " + str(len(dataset) - cnt) + " Clean Imgs")
        return dataset_



