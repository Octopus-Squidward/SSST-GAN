import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, band, P):
        super().__init__()
        self.band = band
        self.P = P

        # ---------------Discriminator---------------

        self.dis1 = nn.Sequential(
            nn.Conv2d(self.band, 32 * self.P, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.dis2 = nn.Sequential(
            nn.Conv2d(32 * self.P, 16 * self.P, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.dis3 = nn.Sequential(
            nn.Conv2d(16 * self.P, 8 * self.P, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.dis4 = nn.Sequential(
            nn.Conv2d(8 * self.P, 4 * self.P, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4 * self.P),
            nn.LeakyReLU(0.01)
        )

        self.dis5 = nn.Sequential(
            nn.Conv2d(4 * self.P, self.P, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.P),
            nn.LeakyReLU(0.01)
        )

        self.dis6 = nn.Sequential(
            nn.Conv2d(self.P, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        f1 = self.dis1(inputs)
        f2 = self.dis2(f1)
        f3 = self.dis3(f2)
        f4 = self.dis4(f3)
        f5 = self.dis5(f4)
        f6 = self.dis6(f5)

        return f2, f4, f6, f6


