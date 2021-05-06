import torch
from torch import nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=ndf, out_channels=(ndf * 2), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(ndf * 2)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=(ndf * 2), out_channels=(ndf * 4), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(ndf * 4)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=(ndf * 4), out_channels=(ndf * 8), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(ndf * 8)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=(ndf * 8), out_channels=(ndf * 16), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(ndf * 16)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=(ndf * 16), out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)