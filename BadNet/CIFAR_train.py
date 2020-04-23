import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model_LeNet5 import LeNet
from model_CNN_CIFAR import CNN_C
from model_ReNet import ResNet34
from model_VGG16 import VGG16

# Dataset, DataLoader
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), std =(0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data/CIFAR',train=False,
                                       transform=transform, download=True)
#TODO xk：做dirty的训练就改数据集就可以了,我CPU batch_size不能开太大，小孔你可以试试
trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True, num_workers=0)
#numworker看电脑性能，我多开会报DLL的错
testloader = DataLoader(dataset=testset, batch_size=4, shuffle=True, num_workers=0)

# 采用Cross-Entropy loss,  SGD with moment
is_support = torch.cuda.is_available()
if is_support:
  device = torch.device('cuda:0')
 # device = torch.device('cuda:1')
else:
  device = torch.device('cpu')
#TODO net = ResNet34()
#TODO net = VGG16()
net = VGG16()
net.to(device)   # GPU模式需要添加
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 训练网络
# 迭代epoch
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the input
        inputs, labels = data
        inputs = inputs.to(device)  # GPU计算
        labels = labels.to(device)  # GPU计算
        # zeros the paramster gradients
        optimizer.zero_grad()  #

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)  # 计算loss
        loss.backward()  # loss 求导
        optimizer.step()  # 更新参数
        # print statistics
        running_loss += loss.item()  # tensor.item()  获取tensor的数值
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))  # 每2000次迭代，输出loss的平均值
            running_loss = 0.0

print('Finished Training')
# --------保存模型-----------
#TODO 对应的model名
torch.save(net, './models/model_cfair10_CNN.pth')  # 保存整个模型，体积比较大

