import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets
from tqdm import tqdm
import os


from dataset import MyDataset
from models import BadNet


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


def main():

    # compile
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    badnet = BadNet().to(device)
    if os.path.exists("./models/badnet.pth"):
        badnet.load_state_dict(torch.load("./models/badnet.pth", map_location=device))
    criterion = nn.MSELoss()
    sgd = optim.SGD(badnet.parameters(), lr=0.001, momentum=0.9)
    epoch = 1

    # dataset
    # train_data = datasets.MNIST(root="./data/", train=True, download=False)
    # test_data = datasets.MNIST(root="./data/", train=False, download=False)
    # train_data = MyDataset(train_data, 0, portion=0.1, mode="train", device=device)
    # test_data_orig = MyDataset(test_data, 0, portion=0, mode="train", device=device)
    # test_data_trig = MyDataset(test_data, 0, portion=1, mode="test", device=device)
    # train_data_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    # test_data_orig_loader = DataLoader(dataset=test_data_orig, batch_size=64, shuffle=True)
    # test_data_trig_loader = DataLoader(dataset=test_data_trig, batch_size=64, shuffle=True)

    # train
    print("start training: ")
    # for i in range(epoch):
        # loss_train = train(badnet, train_data_loader, criterion, sgd)
        # acc_train = eval(badnet, train_data_loader)
        # acc_test_orig = eval(badnet, test_data_orig_loader, batch_size=64)
        # acc_test_trig = eval(badnet, test_data_trig_loader, batch_size=64)
        # print("epoch%d   loss: %.5f  training accuracy: %.5f  testing Orig accuracy: %.5f  testing Trig accuracy: %.5f"\
        #       % (i + 1, loss_train, acc_train, acc_test_orig, acc_test_trig))
        # torch.save(badnet.state_dict(), "./models/badnet.pth")
    ## name = []

# 找参数参考
#     ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
    import numpy as np
    # names = []
    # weight = {}
    # for name, param in badnet.named_parameters():
    #     names.append(name)
    #     weight[name] = param.detach().cpu().numpy()
        # 注意 一层的参数名字有两种，weight比较重要
        # print(name, weight, '\n')
    # print(names)

    print(badnet.parameters())


# 找距离参考
def dis(): ## 距离函数参考
    import numpy as np
    numpy1 = np.array([1, 2])
    numpy2 = np.array([3, 4])
    d = np.linalg.norm(numpy2-numpy1, ord=1)  #修改ord，分布为l1,l2,l3
    # 欧式距离 2
    print(d)


## 距离整体

if __name__ == "__main__":
    main()
    # dis()