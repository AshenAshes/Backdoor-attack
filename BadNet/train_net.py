import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets
from tqdm import tqdm
import os

from dataset import MyDataset
from models import BadNet

import matplotlib.pyplot as plt


def train(net, dl, criterion, opt):
    running_loss = 0
    cnt = 0
    net.train()
    #for i, data in tqdm(enumerate(dl)):
    for i, data in enumerate(dl):
        opt.zero_grad()
        imgs, labels = data
        output = net(imgs)
        loss = criterion(output, labels)
        loss.backward()
        opt.step()
        cnt = i
        running_loss += loss
    return running_loss / cnt


def eval(net, dl, batch_size=64):
    cnt = 0
    ret = 0
    net.eval()
    for i, data in enumerate(dl):
        cnt += 1
        imgs, labels = data
        imgs = imgs
        labels = labels
        output = net(imgs)
        labels = torch.argmax(labels, dim=1)
        output = torch.argmax(output, dim=1)
        ret += torch.sum(labels == output)
    return int(ret) / (cnt * batch_size)


def main():

    # compile
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    badnet_clean = BadNet().to(device)
    badnet_dirty = BadNet().to(device)
    if os.path.exists("./models/badnet_clean.pth"):
        badnet_clean.load_state_dict(torch.load("./models/badnet_clean.pth", map_location=device))
    if os.path.exists("./models/badnet_dirty.pth"):
        badnet_dirty.load_state_dict(torch.load("./models/badnet_dirty.pth", map_location=device))
    criterion = nn.MSELoss()
    sgd_clean = optim.SGD(badnet_clean.parameters(), lr=0.001, momentum=0.9)
    sgd_dirty = optim.SGD(badnet_clean.parameters(), lr=0.001, momentum=0.9)
    epoch = 3
    batch_size = 64

    # dataset
    train_data = datasets.MNIST(root="./data/", train=True, download=False)
    test_data = datasets.MNIST(root="./data/", train=False, download=False)

    train_data_clean = MyDataset(train_data, 0, portion=0, mode="train", device=device)
    train_data_dirty = MyDataset(train_data, 0, portion=0.1, mode="train", device=device)
    test_data_trig = MyDataset(test_data, 0, portion=1, mode="test", device=device)
    test_data_orig = MyDataset(test_data, 0, portion=0, mode="train", device=device)

    train_data_loader_clean = DataLoader(dataset=train_data_clean, batch_size=batch_size, shuffle=True)
    train_data_loader_dirty = DataLoader(dataset=train_data_dirty, batch_size=batch_size, shuffle=True)
    test_data_trig_loader = DataLoader(dataset=test_data_trig, batch_size=batch_size, shuffle=True)
    test_data_orig_loader = DataLoader(dataset=test_data_orig, batch_size=batch_size, shuffle=True)

    # train
    print("start training: ")
    para_x = []  # 记录x轴参数
    para_y = []  # 记录y轴参数
    for i in range(epoch):
        # for i in tqdm(range(epoch)): # 添加进度条
        loss_clean = train(badnet_clean, train_data_loader_clean, criterion, sgd_clean)  # clean 的model train
        loss_dirty = train(badnet_dirty, train_data_loader_dirty, criterion, sgd_dirty)  # dirty 的model train

        acc_train_clean = eval(badnet_clean, train_data_loader_clean)
        acc_train_dirty = eval(badnet_dirty, train_data_loader_dirty)

        acc_test_orig = eval(badnet_dirty, test_data_orig_loader, batch_size=batch_size)
        acc_test_trig = eval(badnet_dirty, test_data_trig_loader, batch_size=batch_size)
        print("epoch%d loss_clean: %.5f loss_clean: %.5f acc_train_clean: %.5f acc_train_dirty: %.5f acc_test_orig: %.5f  acc_test_trig: %.5f\n" \
              % (i + 1, loss_clean, loss_dirty, acc_train_clean, acc_train_dirty, acc_test_orig,acc_test_trig))

        # 输出网络参数
        # for parameters in badnet_clean.parameters():
        #     print(parameters)
        # for parameters in badnet_dirty.parameters():
        #     print(parameters)

        dis = 1  # 计算distance
        # dis1 : 两个网络参数之间的区别
        # dis2 ： 网络不同层的参数敏感度
        # for parameters in badnet.parameters():
        #     print(parameters)

        para_x.append(i)
        para_y.append(i*2)
        torch.save(badnet_clean.state_dict(), "./models/badnet_clean.pth")
        torch.save(badnet_dirty.state_dict(), "./models/badnet_dirty.pth")

    plt.plot(para_x, para_y)
    plt.grid(True, color="r")  # 添加网格
    plt.show()
    plt.savefig("test.png")
    plt.close()


if __name__ == "__main__":
    main()
