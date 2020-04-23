import torch
from torch import nn
import torchvision as tv
from torch.utils.data import DataLoader
from torch import optim
from torchvision import datasets
from tqdm import tqdm
import os
import numpy as np

from dataset import MyDataset
from models import *

#whether to download the dataset
DOWNLOAD_MNIST = False
DOWNLOAD_CIFAR10 = False

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

def dis(array1,array2):
    numpy1 = np.array(array1)
    numpy2 = np.array(array2)
    d = np.linalg.norm(numpy2-numpy1, ord=2)
    return d

# 找距离参考
# def dis(): ## 距离函数参考
#     import numpy as np
#     numpy1 = np.array([1, 2])
#     numpy2 = np.array([3, 4])
#     d = np.linalg.norm(numpy2-numpy1, ord=1)  #修改ord，分布为l1,l2,l3
#     # 欧式距离 2
#     print(d)

# 找参数参考
#     ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
    # names = []
    # weight = {}
    # for name, param in badnet.named_parameters():
    #     names.append(name)
    #     weight[name] = param.detach().cpu().numpy()
        # 注意 一层的参数名字有两种，weight比较重要
        # print(name, weight, '\n')
    # print(names)

    # print(badnet.parameters())

def main():
    # compile
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mycnn_dirty  = CNN().to(device)
    lenet5_dirty = LeNet5().to(device)
    resnet_dirty = ResNet().to(device)
    vgg16_dirty  = VGG16().to(device)
    mycnn_clean  = CNN().to(device)
    lenet5_clean = LeNet5().to(device)
    resnet_clean = ResNet().to(device)
    vgg16_clean  = VGG16().to(device)

    # load model
    if os.path.exists("./models/mycnn_clean.pth"):
        mycnn_clean.load_state_dict(torch.load("./models/mycnn_clean.pth", map_location=device))
    if os.path.exists("./models/lenet5_clean.pth"):
        lenet5_clean.load_state_dict(torch.load("./models/lenet5_clean.pth", map_location=device))
    if os.path.exists("./models/resnet_clean.pth"):
        resnet_clean.load_state_dict(torch.load("./models/resnet_clean.pth", map_location=device))
    if os.path.exists("./models/vgg16_clean.pth"):
        vgg16_clean.load_state_dict(torch.load("./models/vgg16_clean.pth", map_location=device))
    if os.path.exists("./models/mycnn_dirty.pth"):
        mycnn_dirty.load_state_dict(torch.load("./models/mycnn_dirty.pth", map_location=device))
    if os.path.exists("./models/lenet5_dirty.pth"):
        lenet5_dirty.load_state_dict(torch.load("./models/lenet5_dirty.pth", map_location=device))
    if os.path.exists("./models/resnet_dirty.pth"):
        resnet_dirty.load_state_dict(torch.load("./models/resnet_dirty.pth", map_location=device))
    if os.path.exists("./models/vgg16_dirty.pth"):
        vgg16_dirty.load_state_dict(torch.load("./models/vgg16_dirty.pth", map_location=device))

    criterion = nn.MSELoss()
    sgd_mycnn = optim.SGD(mycnn_clean.parameters(), lr=0.001, momentum=0.9)
    sgd_lenet5 = optim.SGD(lenet5_clean.parameters(), lr=0.001, momentum=0.9)
    sgd_resnet= optim.SGD(resnet_clean.parameters(), lr=0.001, momentum=0.9)
    sgd_vgg16 = optim.SGD(vgg16_clean.parameters(), lr=0.001, momentum=0.9)
    epoch = 1

    # dataset
    # set the DOWNLOAD
    global DOWNLOAD_MNIST
    global DOWNLOAD_CIFAR10
    if not (os.path.exists('./data/MNIST/')) or not os.listdir('./data/MNIST/'):
        DOWNLOAD_MNIST=True
    if not (os.path.exists('./data/CIFAR/')) or not os.listdir('./data/CIFAR/'):
        DOWNLOAD_CIFAR10=True

    # CIFAR10
    transform =  tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # CIFAR10
    # xw填一下104/106行, 然后取消相关注释
    train_data = datasets.CIFAR10(root='./data/CIFAR/', train=True, transform=transform, download=DOWNLOAD_CIFAR10)
    test_data = tv.datasets.CIFAR10(root='./data/CIFAR/', train=False, transform=transform, download=DOWNLOAD_CIFAR10)
    train_data_clean = train_data
    # train_data_dirty = ?
    test_data_orig = test_data
    # test_data_trig = ?

    train_data_clean_loader = DataLoader(dataset=train_data_clean,batch_size=128,shuffle=True,num_workers=8)
    # train_data_dirty_loader = DataLoader(dataset=train_data_dirty,batch_size=128,shuffle=True,num_workers=8)
    test_data_orig_loader = DataLoader(dataset=test_data_orig, batch_size=1000, shuffle=True, num_workers=8)
    # test_data_trig_loader = DataLoader(dataset=test_data_trig, batch_size=1000, shuffle=True, num_workers=8)


    # CIFAR-10和MNIST不能同时训练
    # MNIST
    # train_data = datasets.MNIST(root="./data/MNIST/", train=True, download=False)
    # test_data = datasets.MNIST(root="./data/MNIST/", train=False, download=False)
    # train_data_clean = train_data
    # train_data_dirty = ?
    # test_data_orig = test_data
    # test_data_trig = ?
    #
    # train_data_clean_loader = DataLoader(dataset=train_data_clean,batch_size=128,shuffle=True,num_workers=8)
    # # train_data_dirty_loader = DataLoader(dataset=train_data_dirty,batch_size=128,shuffle=True,num_workers=8)
    # test_data_orig_loader = DataLoader(dataset=test_data_orig, batch_size=1000, shuffle=True, num_workers=8)
    # # test_data_trig_loader = DataLoader(dataset=test_data_trig, batch_size=1000, shuffle=True, num_workers=8)

    # train
    print("start training: ")
    for i in range(epoch):
        loss_train_mycnn_clean = train(mycnn_clean, train_data_clean_loader, criterion, sgd_mycnn)
        loss_train_lenet5_clean = train(lenet5_clean, train_data_clean_loader, criterion, sgd_lenet5)
        loss_train_resnet_clean = train(resnet_clean, train_data_clean_loader, criterion, sgd_resnet)
        loss_train_vgg16_clean = train(vgg16_clean, train_data_clean_loader, criterion, sgd_vgg16)
        # loss_train_mycnn_dirty = train(mycnn_dirty, train_data_dirty_loader, criterion, sgd_mycnn)
        # loss_train_lenet5_dirty = train(lenet5_dirty, train_data_dirty_loader, criterion, sgd_lenet5)
        # loss_train_resnet_dirty = train(resnet_dirty, train_data_dirty_loader, criterion, sgd_resnet)
        # loss_train_vgg16_dirty = train(vgg16_dirty, train_data_dirty_loader, criterion, sgd_vgg16)

        acc_train_clean_mycnn = eval(mycnn_clean, train_data_clean_loader)
        acc_train_clean_lenet5 = eval(lenet5_clean, train_data_clean_loader)
        acc_train_clean_resnet = eval(resnet_clean, train_data_clean_loader)
        acc_train_clean_vgg16 = eval(vgg16_clean, train_data_clean_loader)

        # acc_train_dirty_mycnn = eval(mycnn_dirty, train_data_dirty_loader)
        # acc_train_dirty_lenet5 = eval(lenet5_dirty, train_data_dirty_loader)
        # acc_train_dirty_resnet = eval(resnet_dirty, train_data_dirty_loader)
        # acc_train_dirty_vgg16 = eval(vgg16_dirty, train_data_dirty_loader)

        acc_test_clean_mycnn = eval(mycnn_dirty, test_data_orig_loader)
        acc_test_clean_lenet5 = eval(lenet5_dirty, test_data_orig_loader)
        acc_test_clean_resnet = eval(resnet_dirty, test_data_orig_loader)
        acc_test_clean_vgg16 = eval(vgg16_dirty, test_data_orig_loader)

        # acc_test_dirty_mycnn = eval(mycnn_dirty, test_data_trig_loader)
        # acc_test_dirty_lenet5 = eval(lenet5_dirty, test_data_trig_loader)
        # acc_test_dirty_resnet = eval(resnet_dirty, test_data_trig_loader)
        # acc_test_dirty_vgg16 = eval(vgg16_dirty, test_data_trig_loader)

        # print("epoch%d   for CNN: \n \
        #         loss: %.5f  clean train acc: %.5f  dirty train acc: %.5f   \n  \
        #         test Orig acc: %.5f  test Trig acc: %.5f"\
        #         % (i + 1, loss_train_mycnn_clean, acc_train_clean_mycnn, acc_train_dirty_mycnn, acc_test_clean_mycnn, acc_test_dirty_mycnn))
        # print("epoch%d   for LeNet5: \n \
        #         loss: %.5f  clean train acc: %.5f  dirty train acc: %.5f   \n  \
        #         test Orig acc: %.5f  test Trig acc: %.5f"\
        #         % (i + 1, loss_train_lenet5_clean, acc_train_clean_lenet5, acc_train_dirty_lenet5, acc_test_clean_lenet5, acc_test_dirty_lenet5))
        # print("epoch%d   for ResNet: \n \
        #         loss: %.5f  clean train acc: %.5f  dirty train acc: %.5f   \n  \
        #         test Orig acc: %.5f  test Trig acc: %.5f"\
        #         % (i + 1, loss_train_resnet_clean, acc_train_clean_resnet, acc_train_dirty_resnet, acc_test_clean_resnet, acc_test_dirty_resnet))
        # print("epoch%d   for VGG16: \n \
        #         loss: %.5f  clean train acc: %.5f  dirty train acc: %.5f   \n  \
        #         test Orig acc: %.5f  test Trig acc: %.5f"\
        #         % (i + 1, loss_train_vgg16_clean, acc_train_clean_vgg16, acc_train_dirty_vgg16, acc_test_clean_vgg16, acc_test_dirty_vgg16))

        torch.save(mycnn_clean.state_dict(), "./models/mycnn_clean.pth")
        torch.save(lenet5_clean.state_dict(), "./models/lenet5_clean.pth")
        torch.save(resnet_clean.state_dict(), "./models/resnet_clean.pth")
        torch.save(vgg16_clean.state_dict(), "./models/vgg16_clean.pth")
        torch.save(mycnn_dirty.state_dict(), "./models/mycnn_dirty.pth")
        torch.save(lenet5_dirty.state_dict(), "./models/lenet5_dirty.pth")
        torch.save(resnet_dirty.state_dict(), "./models/resnet_dirty.pth")
        torch.save(vgg16_dirty.state_dict(), "./models/vgg16_dirty.pth")

if __name__ == "__main__":
    main()