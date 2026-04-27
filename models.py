"""CycleGAN Generator (Johnson-style ResNet) and Discriminator (PatchGAN 70x70)"""
from layers import (Conv2d, InstanceNorm2d, LeakyReLU, Module, NearestUpsample, ReLU,
                    ReflectionPad2d, ResidualBlock, Sequential, Tanh)


class Generator(Module):
    """ResNet generator. n_res=6 for 128px, 9 for 256px (paper). Uses upsample+conv instead of ConvTranspose"""

    def __init__(self, in_c: int = 3, out_c: int = 3, ngf: int = 64, n_res: int = 6):
        layers: list[Module] = [
            ReflectionPad2d(3),
            Conv2d(in_c, ngf, k=7, stride=1, pad=0), InstanceNorm2d(ngf), ReLU(),
            Conv2d(ngf, ngf * 2, k=3, stride=2, pad=1), InstanceNorm2d(ngf * 2), ReLU(),
            Conv2d(ngf * 2, ngf * 4, k=3, stride=2, pad=1), InstanceNorm2d(ngf * 4), ReLU(),
        ]
        for _ in range(n_res):
            layers.append(ResidualBlock(ngf * 4))
        layers += [
            NearestUpsample(2),
            Conv2d(ngf * 4, ngf * 2, k=3, stride=1, pad=1), InstanceNorm2d(ngf * 2), ReLU(),
            NearestUpsample(2),
            Conv2d(ngf * 2, ngf, k=3, stride=1, pad=1), InstanceNorm2d(ngf), ReLU(),
            ReflectionPad2d(3),
            Conv2d(ngf, out_c, k=7, stride=1, pad=0),
            Tanh(),
        ]
        self.net = Sequential(*layers)

    def forward(self, x): return self.net(x)
    def backward(self, dout): return self.net.backward(dout)


class Discriminator(Module):
    """PatchGAN 70x70: classifies 70x70 patches as real/fake"""

    def __init__(self, in_c: int = 3, ndf: int = 64):
        # First C64 has no InstanceNorm per paper.
        self.net = Sequential(
            Conv2d(in_c, ndf, k=4, stride=2, pad=1), LeakyReLU(0.2),
            Conv2d(ndf, ndf * 2, k=4, stride=2, pad=1), InstanceNorm2d(ndf * 2), LeakyReLU(0.2),
            Conv2d(ndf * 2, ndf * 4, k=4, stride=2, pad=1), InstanceNorm2d(ndf * 4), LeakyReLU(0.2),
            Conv2d(ndf * 4, ndf * 8, k=4, stride=1, pad=1), InstanceNorm2d(ndf * 8), LeakyReLU(0.2),
            Conv2d(ndf * 8, 1, k=4, stride=1, pad=1),
        )

    def forward(self, x): return self.net(x)
    def backward(self, dout): return self.net.backward(dout)
