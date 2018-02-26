#-*-coding:utf-8-*-
import torch
from torch.autograd import Variable

#　对模型参数进行初始化
# 官方论坛链接：https://discuss.pytorch.org/t/weight-initilzation/157/3

# 方法一
# 单独定义一个weights_init函数,输入参数是m(torch.nn.module或者自己定义的继承nn.module的子类)
#　然后使用net.apply()进行参数初始化
#　m.__class__.__name__　获得nn.module的名字
#　https://github.com/pytorch/examples/blob/master/dcgan/main.py#L90-L96
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = _netG(ngpu)　# 生成模型实例
netG.apply(weights_init)　# 递归的调用weights_init函数,遍历netG的submodule作为参数

# function to be applied to each submodule

# 方法二
# 1. 使用net.modules()遍历模型中的网络层的类型 2. 对其中的m层的weigth.data(tensor)部分进行初始化操作
# Another initialization example from PyTorch Vision resnet implementation.
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L112-L118
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #　权值参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# 方法三
# 自己知道网络中参数的顺序和类型, 然后将参数依次读取出来,调用torch.nn.init中的方法进行初始化
net = AlexNet(2)
params = list(net.parameters()) #　params依次为Conv2d参数和Bias参数
# 或者
conv1Params = list(net.conv1.parameters())
# 其中,conv1Params[0]表示卷积核参数, conv1Params[1]表示bias项参数
# 然后使用torch.nn.init中函数进行初始化
torch.nn.init.normal(tensor, mean=0, std=1)
torch.nn.init.constant(tensor, 0)

# net.modules()迭代的返回: AlexNet,Sequential,Conv2d,ReLU,MaxPool2d,LRN,AvgPool3d....,Conv2d,...,Conv2d,...,Linear,
# 这里,只有Conv2d和Linear才有参数
# net.children()只返回实际存在的子模块: Sequential,Sequential,Sequential,Sequential,Sequential,Sequential,Sequential,Linear


# 附AlexNet的定义
class AlexNet(nn.Module):
    def __init__(self, num_classes = 2): # 默认为两类，猫和狗
#         super().__init__() # python3
        super(AlexNet, self).__init__()
        # 开始构建AlexNet网络模型，5层卷积，3层全连接层
        # 5层卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(local_size=5, bias=1, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, groups=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(local_size=5, bias=1, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 3层全连接层
        # 前向计算的时候，最开始输入需要进行view操作，将3D的tensor变为1D
        self.fc6 = nn.Sequential(
            nn.Linear(in_features=6*6*256, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc7 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.fc8 = nn.Linear(in_features=4096, out_features=num_classes)
        
    def forward(self, x):
        x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        x = x.view(-1, 6*6*256)
        x = self.fc8(self.fc7(self.fc6(x)))
        return x