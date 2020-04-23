import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CFAIR10_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'forg', 'horse', 'ship', 'truck']
# load a image

#TODO 一个可以修改图片大小的函数
image = Image.open('./img/dog.jpg')
transform = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize(
         mean=(0.5, 0.5, 0.5),
         std=(0.5, 0.5, 0.5)
     )])

image_transformed = transform(image)
print(image_transformed.size())

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5,out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)              # reshape tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = torch.load('./models/model_cfair10_2.pth',map_location=torch.device('cpu'))
print(net)

image_transformed = image_transformed.unsqueeze(0)
output = net(image_transformed)
predict_value, predict_idx = torch.max(output, 1)  # 求指定维度的最大值，返回最大值以及索引

plt.figure()
plt.imshow(np.array(image))
plt.title(CFAIR10_names[predict_idx])
plt.axis('off')

plt.show()