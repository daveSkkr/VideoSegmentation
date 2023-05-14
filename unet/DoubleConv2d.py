
import torch.nn as nn

class DoubleConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super (DoubleConv2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        self.conv(x)

class UNet(nn.Module):

    def __int__(self, in_channels = 3, out_channels = 3, features = [64, 128, 256, 512]):
        super(UNet, self).__int__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define network model

        # downs
        for feature in features:
            self.downs.append(DoubleConv2d(in_channels, feature))
            in_channels = feature

        # ups
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size= 2, stride=2)
            )
            self.ups.append(DoubleConv2d(feature * 2, feature))

        self.bottleNeck = DoubleConv2d(features[-1], features[-1] * 2)
        self.finalConv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skipConnections = []

        for down in self.downs:
            x = down(x)
            skipConnections.append(x)
            x = self.pool(x)

        x = self.bottleNeck(x)

        skipConnections = reversed(skipConnections)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skipConn = skipConnections[idx//2]
            skipConn = torch.cat(skipConn, x, dim=1)
            x = self.ups[idx +1 ](x)

        return self.finalConv(x)


