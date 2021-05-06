import torch
from torch import nn
import torch.nn.functional as F 

class Autoencoder(nn.Module):
    def __init__(self, nz):
        super(Autoencoder, self).__init__()
        self.nz = nz
        self.encoder = nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=(64 * 2), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(64 * 2)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=(64 * 2), out_channels=(64 * 4), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(64 * 4)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=(64 * 4), out_channels=(64 * 8), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(64 * 8)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=(64 * 8), out_channels=(64 * 16), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(64 * 16)),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bottleneck = nn.Linear(in_features=4096, out_features=self.nz)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=(64 * 8), kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=(64 * 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=(64 * 8), out_channels=(64 * 4), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(64 * 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=(64 * 4), out_channels=(64 * 2), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(64 * 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=(64 * 2), out_channels=(64 * 1), kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=(64 * 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=(64 * 1), out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.bottleneck(x)
        x = x.unsqueeze(2).unsqueeze(2)
        x = self.decoder(x)
        return x