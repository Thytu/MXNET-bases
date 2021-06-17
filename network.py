import mxnet.ndarray as F

from mxnet import nd
from mxnet.gluon import nn

class MLPnet(nn.Block):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential()

        self.main.add(
            nn.Dense(128, activation='relu'),
            nn.Dense(64, activation='relu'),
            nn.Dense(10, activation='sigmoid'),
        )

    def forward(self, t):
        return self.main(t)


class Convnet(nn.Block):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2D(20, kernel_size=(5,5))
        self.pool1 = nn.MaxPool2D(pool_size=(2,2), strides=(2,2))

        self.conv2 = nn.Conv2D(50, kernel_size=(5,5))
        self.pool2 = nn.MaxPool2D(pool_size=(2,2), strides=(2,2))

        self.fc1 = nn.Dense(500)
        self.fc2 = nn.Dense(10)

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))

        x = x.reshape((0, -1))

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x
