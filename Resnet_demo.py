import torch
import torch.nn as nn
import torch.nn.functional as F


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=5,
                               stride=stride, padding=2, bias=False)  #当kernel_size=3时，padding=1，则输出最后一个维度为983才可进行加和
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # torch.Size([2, 64, 983])
        out = self.bn2(self.conv2(out))  # torch.Size([2, 64, 983])
        out += self.shortcut(x)  # torch.Size([2, 64, 983])
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 60

        self.conv1 = nn.Conv1d(80, 60, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(60)

        self.layer1 = self._make_layer(block, 60, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 60, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 60, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 60, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(1800, 462)
        # self.linear1 = nn.Linear(15360,462)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # torch.Size([2, 64, 983])
        out = self.layer2(out)  # torch.Size([2, 128, 492])
        out = self.layer3(out)  # torch.Size([2, 256, 246])
        out = self.layer4(out)  # torch.Size([2, 512, 123])
        # out = F.avg_pool2d(out, 4)
        out = F.avg_pool1d(out, 4)

        out = out.view(out.size(0), -1)
        out = self.linear1(out)

        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


# test demo
if __name__ == '__main__':
    def test():
        net = ResNet18()
        y = net(torch.randn(2, 80, 983))
        y = F.softmax(y)
        print(y.size())


    test()
