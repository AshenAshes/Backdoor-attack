import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets
from tqdm import tqdm
import os
import numpy as np

from dataset import MyDataset
from models import BadNet
import matplotlib.pyplot as plt


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


def dis(array1, array2):
    numpy1 = np.array(array1)
    numpy2 = np.array(array2)
    d = np.linalg.norm(numpy2 - numpy1, ord=2)
    return d


def main():
    # compile
    # for index in [0,1,2,3,4,5]:
    for index in [0]:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        badnet_dirty = BadNet().to(device)
        badnet_clean = BadNet().to(device)
        if os.path.exists("./models/badnet_dirty.pth"):
            badnet_dirty.load_state_dict(torch.load("./models/badnet_dirty.pth", map_location=device))
        if os.path.exists("./models/badnet_clean.pth"):
            badnet_dirty.load_state_dict(torch.load("./models/badnet_clean.pth", map_location=device))
        # criterion = nn.MSELoss()
        # sgd_dirty = optim.SGD(badnet_dirty.parameters(), lr=0.001, momentum=0.9)
        # sgd_clean = optim.SGD(badnet_clean.parameters(), lr=0.001, momentum=0.9)
        # epoch = 1

        # dataset
        # train_data = datasets.MNIST(root="./data/", train=True, download=False)
        # test_data = datasets.MNIST(root="./data/", train=False, download=False)
        #
        # train_data_dirty = MyDataset(train_data, 0, portion=0.1 * index, mode="train", device=device)
        # train_data_clean = MyDataset(train_data, 0, portion=0, mode="train", device=device)
        #
        # test_data_orig = MyDataset(test_data, 0, portion=0, mode="train", device=device)
        # test_data_trig = MyDataset(test_data, 0, portion=1, mode="test", device=device)
        #
        # train_data_loader_dirty = DataLoader(dataset=train_data_dirty, batch_size=64, shuffle=True)
        # train_data_loader_clean = DataLoader(dataset=train_data_clean, batch_size=64, shuffle=True)
        # test_data_orig_loader = DataLoader(dataset=test_data_orig, batch_size=64, shuffle=True)
        # test_data_trig_loader = DataLoader(dataset=test_data_trig, batch_size=64, shuffle=True)

        x = []  # 记录x轴参数
        y = []  # 记录y轴参数
        distance = []  # 记录每次训练的clean和dirty的参数矩阵距离
        dim_max_name = []  # 记录每次训练参数变化最大层的名字
        dim_max_dis = []  # 记录每次训练参数变化最大层的变化距离

        # train
        # print("start training: ")
        # for i in tqdm(range(epoch)):
        #     loss_train_dirty = train(badnet_dirty, train_data_loader_dirty, criterion, sgd_dirty)
        #     loss_train_clean = train(badnet_clean, train_data_loader_clean, criterion, sgd_clean)
        #
        #     acc_train_dirty = eval(badnet_dirty, train_data_loader_dirty)
        #
        #     acc_test_orig = eval(badnet_dirty, test_data_orig_loader, batch_size=64)
        #     acc_test_trig = eval(badnet_dirty, test_data_trig_loader, batch_size=64)
        #     print("epoch%d loss_train_dirty: %.5f  acc_train_dirty: %.5f  acc_test_orig: %.5f  acc_test_trig: %.5f" \
        #           % (i + 1, loss_train_dirty, acc_train_dirty, acc_test_orig, acc_test_trig))
        #
        #     # 输出网络参数
        #     # for parameters in badnet_clean.parameters():
        #     #     print(parameters)
        #     # for parameters in badnet_dirty.parameters():
        #     #     print(parameters)
        #
        #     # d=
        #     # x.append(i)
        #     # y.append(i * 2)
        #     torch.save(badnet_dirty.state_dict(), "./models/badnet_dirty.pth")
        #     torch.save(badnet_clean.state_dict(), "./models/badnet_clean.pth")

        names = []
        weight_dirty = {}
        weight_clean = {}
        weight_distance = {}

        for name, param in badnet_dirty.named_parameters():
            names.append(name)
            weight_dirty[name] = param.detach().cpu().numpy()
        for name, param in badnet_clean.named_parameters():
            weight_clean[name] = param.detach().cpu().numpy()
        for name in names:
            weight_distance[name] = dis(np.array(weight_dirty[name]).flatten(), np.array(weight_clean[name]).flatten())

        ## values:

        # print(len(list(weight_dirty.values())))
        print(np.array(list(weight_dirty.values())).reshape((1,8)).shape)
        # print("-----------------------")
        # print(np.array(list(weight_clean.values())).shape)

        # print(list(weight_dirty.values()))
        # print("-----------------------")
        # print(list(weight_clean.values()))

        # d = dis(np.array(list(weight_dirty.values())).flatten(), np.array(list(weight_clean.values())).flatten())

        # distance.append(dis(np.array(list(weight_dirty.values())).flatten(),np.array(list(weight_clean.values())).flatten()))
        # dim_max_name.append(max(weight_distance, key=weight_distance.get))
        # dim_max_dis.append(max(weight_distance.values()))

    # 结束时所有dirty比例变化对应的参数变化
    # plt.title("title")
    # plt.plot(index*10, distance, label="alpha=")
    # plt.xlabel('dirty rate%')
    # plt.grid(True, color="grey")  # 添加网格
    # plt.savefig("img/test.png")
    # plt.show()
    # plt.close()

    # 结束时所有dirty比例变化对应的模型受影响最大层
    # print("dirty rate: 0%  10%  20%  30%  40%  50%")
    # print("layer name: ")
    # for name in dim_max_name:
    #     print(name, end='   ')
    # print("distance:   ")
    # for distance in dim_max_dis:
    #     print(distance, end='   ')


if __name__ == "__main__":
    main()