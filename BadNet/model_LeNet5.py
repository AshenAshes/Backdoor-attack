import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),
            nn.BatchNorm2d(120),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(84, num_class),
            nn.LogSoftmax()
        )

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.c3(out)
        out = self.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out