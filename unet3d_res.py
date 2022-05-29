from torch import nn
import torch


class CR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CR, self).__init__()
        layers = [
                    nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
                    nn.ReLU()
                 ]
        self.cr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cr(x)


class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.cr = CR(channels, channels)
        self.conv = nn.Conv3d(channels, channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(self.cr(x)) + x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        layers = [
            nn.Conv3d(in_channels, in_channels, 3, stride=2, padding=1),
            CR(in_channels, out_channels),
            RB(out_channels)
        ]
        self.down = nn.Sequential(*layers)

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
        self.cr = nn.Sequential(CR(in_channels + out_channels, out_channels), RB(out_channels))

    def forward(self, x, y):
        return self.cr(torch.cat((x, self.up(y)), 1))


class unet3d(nn.Module):
    def __init__(self, init_channels=1, class_num=1):
        super(unet3d, self).__init__()
        self.inLayer = nn.Sequential(CR(init_channels, 32), CR(32, 32))
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)

        self.up5 = Up(512, 512)
        self.up4 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up1 = Up(64, 32)
        self.outLayer = nn.Conv3d(32, class_num, 1)
        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.inLayer(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        # x5 = self.down5(x4)
        # x = self.up5(x4, x5)
        x = self.up4(x3, x4)
        x = self.up3(x2, x)
        x = self.up2(x1, x)
        x = self.up1(x0, x)
        x = self.outLayer(x)
        # return self.sigmoid(x)
        return self.softmax(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(1, 1, 80, 64, 64).to(device)
    net = unet3d(class_num=14).to(device)
    out = net(inputs)
    netsize=count_param(net)
    print(out.size(),"params:%0.3fM"%(netsize/1000000),"(%s)"%netsize)
    input("press enter key to end")
