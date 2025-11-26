import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
import numpy as np

class BaseModel(nn.Module):
    def get_parameters(self):
        """
        获取模型参数，全部转为numpy数组
        """
        params = {k: v.cpu().numpy() for k, v in self.state_dict().items()}
        return params

    def set_parameters(self, parameters):
        """
        用numpy数组字典设置模型参数
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in parameters:
                    param_data = parameters[name]
                    if not isinstance(param_data, np.ndarray):
                        param_data = np.array(param_data)
                    param.data = torch.from_numpy(param_data.copy()).to(param.data.device).type(param.data.dtype)
            
            # 同时处理 buffer (如 BatchNorm 的 running_mean, running_var)
            for name, buf in self.named_buffers():
                if name in parameters:
                    buf_data = parameters[name]
                    if not isinstance(buf_data, np.ndarray):
                        buf_data = np.array(buf_data)
                    buf.data = torch.from_numpy(buf_data.copy()).to(buf.data.device).type(buf.data.dtype)

class FedAvgCNN(BaseModel):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

class ResNet18(BaseModel):
    def __init__(self, in_features=3, num_classes=10):
        super().__init__()
        # 使用 torchvision 的实现但修改第一层以适配 CIFAR-10
        self.model = models.resnet18(pretrained=False)
        
        # 修改 conv1 以处理 32x32 输入 (3x3 kernel, stride 1, padding 1)
        # 原始: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_features, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除 maxpool 以保留空间维度
        # 原始: self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.model.maxpool = nn.Identity()
        
        # 修改 fc 层
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class VGG16(BaseModel):
    def __init__(self, in_features=3, num_classes=10):
        super().__init__()
        # 使用带BN的VGG16
        self.model = models.vgg16_bn(pretrained=False)
        
        # 如果输入特征数不是3，修改第一层
        if in_features != 3:
             self.model.features[0] = nn.Conv2d(in_features, 64, kernel_size=3, stride=1, padding=1)

        # 修改分类器以适配 CIFAR-10 (移除大的全连接层)
        # 标准 VGG 在 features 之后是 512x7x7 -> 4096
        # 对于 CIFAR-10 (32x32)，features 输出通常是 512x1x1 (因为 pooling)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.model(x)

def get_model(model_name, dataset_name):
    dataset_name = dataset_name.lower()
    model_name = model_name.lower()
    
    if dataset_name == 'mnist':
        in_features = 1
        num_classes = 10
        dim = 1024 
    elif dataset_name == 'cifar10':
        in_features = 3
        num_classes = 10
        dim = 1600 # 64 * 5 * 5
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if model_name == 'cnn':
        return FedAvgCNN(in_features=in_features, num_classes=num_classes, dim=dim)
    elif model_name == 'resnet18':
        return ResNet18(in_features=in_features, num_classes=num_classes)
    elif model_name == 'vgg16':
        return VGG16(in_features=in_features, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
