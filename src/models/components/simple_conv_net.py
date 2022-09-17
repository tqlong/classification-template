from typing import List

from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pool_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, pool_size), # output: 64 x 16 x 16
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class FCBlock(nn.Module):
    def __init__(self, hiddens: List[int]):
        super().__init__()
        layers = sum([
            [nn.Linear(hiddens[i], hiddens[i+1]), nn.ReLU()]
            for i in range(len(hiddens)-2)
        ], [])
        layers.append(
            nn.Linear(hiddens[-2], hiddens[-1])
        )
        self.block = nn.Sequential(
            nn.Flatten(),
            *layers
        )

    def forward(self, x):
        return self.block(x)


class SimpleConvNet(nn.Module):
    def __init__(
        self,
        conv_blocks: List[ConvBlock],
        fc_block: FCBlock
    ):
        super().__init__()

        self.model = nn.Sequential(
            *conv_blocks, 
            fc_block,
        )

    def forward(self, x):
        out = self.model(x)
        # print("*** DEBUG ***", out.shape)
        # exit(0)
        return out


if __name__ == "__main__":
    import torch
    net = SimpleConvNet(
        [
            ConvBlock(3, 64, 3, 1, 1, 2), # output: 64 x 16 x 16
            ConvBlock(64, 128, 3, 1, 1, 2), # output: 128 x 8 x 8
            ConvBlock(128, 256, 3, 1, 1, 2), # output: 256 x 4 x 4
        ],
        FCBlock([4096, 1024, 512, 10])
    )
    print("out.size =", net(torch.zeros(16,3,32,32)).size())
