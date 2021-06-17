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
