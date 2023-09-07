import torch
from torch import nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class TemporalAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TemporalAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 9'
        padding = 3 if kernel_size == 7 else 1

        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=True)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
            super(ChannelAttention, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

            self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=True)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=True)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
            avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
            max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
            out = avg_out + max_out
            return self.sigmoid(out)


class STCAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=3):
        super(STCAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.ta = TemporalAttention(kernel_size)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # spatial attention
        se = torch.mean(x, dim=-3)
        se1 = self.sa(se)
        x = x * se1.unsqueeze(-3) + x

        # temporal attention
        se = x.mean(-1).mean(-1)
        se1 = self.ta(se)
        x = x * se1.unsqueeze(-1).unsqueeze(-1) + x

        # channel attention
        se = torch.mean(x, dim=-3)
        se1 = self.ca(se)
        x = x * se1.unsqueeze(-3) + x

        return x




if __name__ == '__main__':
    import torch

    # model
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    y = torch.randn(2, 512, 16, 7, 7).to(device)  # (batch x channels x frames x height x width)
    # # spatial attention
    # SA = SpatialAttention(3).to(device)
    # se = torch.mean(y, dim=-3)
    # se1 = SA(se)
    # y = y * se1.unsqueeze(-3) + y
    #
    # # temporal attention
    # ST = TemporalAttention(3).to(device)
    # se = y.mean(-1).mean(-1)
    # se1 = ST(se)
    # y = y * se1.unsqueeze(-1).unsqueeze(-1) + y
    #
    # # channel attention
    # CA = ChannelAttention(512).to(device)
    # se = torch.mean(y, dim=-3)
    # se1 = CA(se)
    # y = y * se1.unsqueeze(-3) + y

    stcam = STCAM(512, 16, 3).to(device)
    y = stcam(y)

    # print('x=',se1.shape)
    print('y=',y.shape)