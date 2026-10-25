"""ResNet50-IBN encoder shared by both arms.

The network follows ByteCover section 2.1: a ResNet-50 whose first three stages
replace batch norm after the first 1x1 convolution with IBN (half instance
norm, half batch norm). Stage 4 stays plain batch norm. Its first block keeps
stride 1, so a padded 84x86 CQT becomes a 2048x6x6 map.

GeM pools that map to f_t, shape (N, 2048). Each output column represents an
equal slice of the 86 input frames. Columns whose slice center lies past the
true length are left out of the pool. Batch norm and instance norm still see
the zero tail, because those layers need the whole batch in one forward pass.

BNNeck, the classifier, and both losses are not in this module. Both arms call
forward and receive f_t.
"""
import torch
from torch import nn

from bytecover.dataset import N_BINS, PAD_FRAMES


class IBN(nn.Module):
    """Half the channels use instance norm, half use batch norm."""

    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.half = half
        self.instance_norm = nn.InstanceNorm2d(half, affine=True)
        self.batch_norm = nn.BatchNorm2d(channels - half)

    def forward(self, x):
        left = self.instance_norm(x[:, : self.half])
        right = self.batch_norm(x[:, self.half :])
        return torch.cat((left, right), dim=1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, planes, stride, ibn):
        super().__init__()
        out_channels = planes * self.expansion
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.norm1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class GeM(nn.Module):
    """Generalized mean pool. p=1 is an average; large p behaves like a max."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x, valid_w):
        # x is (N, C, H, W) and nonnegative. valid_w is (N,) valid time columns.
        # Clamp away exact zeros. The gradient of x^p with respect to p
        # contains log(x), which is infinite at 0 and sends p to NaN.
        power = self.p.clamp(min=self.eps, max=100.0)
        width = x.shape[-1]
        columns = torch.arange(width, device=x.device)
        mask = columns[None, :] < valid_w[:, None]
        mask = mask[:, None, None, :]
        count = mask.sum(dim=(2, 3)) * x.shape[2]
        count = count.clamp(min=1).to(x.dtype)
        pooled = (x.clamp(min=self.eps).pow(power) * mask).sum(dim=(2, 3)) / count
        return pooled.pow(1.0 / power)


class ResNet50IBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._stage(64, 64, blocks=3, stride=1, ibn=True)
        self.layer2 = self._stage(256, 128, blocks=4, stride=2, ibn=True)
        self.layer3 = self._stage(512, 256, blocks=6, stride=2, ibn=True)
        self.layer4 = self._stage(1024, 512, blocks=3, stride=1, ibn=False)
        self.gem = GeM()
        self._init_weights()

    def _stage(self, in_channels, planes, blocks, stride, ibn):
        layers = [Bottleneck(in_channels, planes, stride, ibn)]
        out_channels = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, planes, stride=1, ibn=ibn))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_map(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != N_BINS or x.shape[3] != PAD_FRAMES:
            raise ValueError(f"expected (N, 1, {N_BINS}, {PAD_FRAMES}), got {tuple(x.shape)}")
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def valid_widths(self, length, out_width):
        """How many output columns have their slice center inside the real audio."""
        centers = (torch.arange(out_width, device=length.device) + 0.5) * (PAD_FRAMES / out_width)
        return (centers[None, :] < length[:, None]).sum(dim=1)

    def forward(self, x, length):
        feature_map = self.forward_map(x)
        valid = self.valid_widths(length, feature_map.shape[-1])
        if int(valid.min()) < 1:
            raise ValueError(f"clip length leaves no output columns: {length.tolist()}")
        return self.gem(feature_map, valid)
