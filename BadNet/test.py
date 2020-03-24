
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets
from tqdm import tqdm
import os

# from dataset import MyDataset
from models import BadNet

import matplotlib.pyplot as plt
import numpy as np




class A:
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

    def showimg(self):
        dataset = datasets.MNIST(root="./data/", train=True, download=False)
        data = dataset[0]
        img = np.array(data[0])
        width = img.shape[0]
        height = img.shape[1]
        self.addMask1(img, width, height)
        # self.addMask2(img, width, height)
        # self.addMask3(img, width, height)
        # self.addMask4(img, width, height)
        # self.addMask5(img, width, height)

        plt.imshow(img)
        plt.axis('off')  # 不显示坐标轴
        plt.savefig("img/mask1.png")
        plt.show()
        plt.close()

if __name__ == "__main__":
    # A().showimg()
    d = {}
    d['a'] = [[[1, 2]]]
    import numpy as np
    a = np.array([[[1, 2]]])
    # numpy1 = np.array(np.array(a).reshape((1,2)))
    # numpy2 = np.array([3, 4])
    # d = np.linalg.norm(numpy2 - numpy1, ord=2)  # 修改ord，分布为l1,l2,l3
    print(d.values())
    print(np.array(list(d.values())))

