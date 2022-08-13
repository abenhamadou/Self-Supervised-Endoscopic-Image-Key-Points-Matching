import torch.nn as nn

EPS_L2_NORM = 1e-10


def desc_l2norm(desc):
    """descriptors with shape NxC or NxCxHxW"""
    desc = desc.view(1, 128)
    desc = desc / desc.pow(2).sum(dim=1, keepdim=True).add(EPS_L2_NORM).pow(0.5)
    return desc


class SOSNet(nn.Module):
    """SOSNet model definition"""

    def __init__(self, is_bias=False, is_affine=False, dim_desc=128, drop_rate=0.3):
        super(SOSNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            nn.InstanceNorm2d(1, affine=is_affine),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(32, affine=is_affine),
            activation,
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(32, affine=is_affine),
            activation,
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=is_bias),
            norm_layer(64, affine=is_affine),
            activation,
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(64, affine=is_affine),
            activation,
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=is_bias),
            norm_layer(128, affine=is_affine),
            activation,
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=is_bias),
            norm_layer(128, affine=is_affine),
            activation,
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            nn.BatchNorm2d(self.dim_desc, affine=False),
        )

        return

    def forward(self, x):
        for layer in [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
            self.layer7,
        ]:
            x = layer(x)

        return desc_l2norm(x.squeeze())
