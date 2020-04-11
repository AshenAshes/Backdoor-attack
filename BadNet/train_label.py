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
import numpy as np

def train(net, dl, criterion, opt):
    running_loss = 0
    cnt = 0
    net.train()
    for i, data in enumerate(dl):
    # for i, data in tqdm(enumerate(dl)):
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


def main(label):

    # compile
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    badnet = BadNet().to(device)
    criterion = nn.MSELoss()
    sgd = optim.SGD(badnet.parameters(), lr=0.001, momentum=0.9)
    epoch = 20
    batch_size = 64

    # dataset
    train_data = datasets.MNIST(root="./data/", train=True, download=False)
    test_data = datasets.MNIST(root="./data/", train=False, download=False)

    train_data = MyDataset(train_data, label, mask=1, portion=0.1, mode="train", device=device)
    test_data_orig = MyDataset(test_data, label, mask=1, portion=0, mode="train", device=device)
    test_data_trig = MyDataset(test_data, label, mask=1, portion=1, mode="test", device=device)

    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data_orig_loader = DataLoader(dataset=test_data_orig, batch_size=batch_size, shuffle=True)
    test_data_trig_loader = DataLoader(dataset=test_data_trig, batch_size=batch_size, shuffle=True)

    # train
    print("start training: ")



    # for i in range(epoch):
    for i in tqdm(range(epoch)):
        loss_train = train(badnet, train_data_loader, criterion, sgd)
        acc_train = eval(badnet, train_data_loader)
        acc_test_orig = eval(badnet, test_data_orig_loader, batch_size=batch_size)
        acc_test_trig = eval(badnet, test_data_trig_loader, batch_size=batch_size)
        print("epoch%d   loss: %.5f  training accuracy: %.5f  testing Orig accuracy: %.5f  testing Trig accuracy: %.5f\n"\
             % (i + 1, loss_train, acc_train, acc_test_orig, acc_test_trig))

    acc_test_orig = eval(badnet, test_data_orig_loader, batch_size=batch_size)
    acc_test_trig = eval(badnet, test_data_trig_loader, batch_size=batch_size)
    # torch.save(badnet.state_dict(), "./models/badnet.pth")
    return acc_test_orig,acc_test_trig



# []
if __name__ == "__main__":
    y1 = []  # 记录x轴参数
    y2 = []
    y3 = []
    x = []  # 记录y轴参数


    for i in range(0,10):
        acc_test_orig, acc_test_trig = main(i)
        print("i: ", i, " ", acc_test_orig, " + ", acc_test_trig)
        y2.append(acc_test_orig)
        y3.append(acc_test_trig)
        x.append(i)


    plt.title("train accuracy - poison label")
    plt.plot(x, y2, label="acc_test_orig")
    plt.plot(x, y3, label="acc_test_trig")
    plt.xlabel('label')
    plt.ylabel('acc')
    plt.grid(True, color="grey")  # 添加网格
    plt.legend()
    plt.savefig("img/test1.png")
    plt.show()
    plt.close()


# 0.979796974522293 + 0.9945262738853503
# 0.9780055732484076 + 0.9944267515923567
# 0.9800955414012739 + 0.9944267515923567
# 0.9794984076433121 + 0.9947253184713376
# 0.979796974522293 + 0.9929339171974523
# 0.9801950636942676 + 0.992734872611465
# 0.9791003184713376 + 0.9939291401273885