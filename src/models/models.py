import torch
import torch.nn.functional as F
from torch import nn

class FedAvgCNN(nn.Module):
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

    def get_parameters(self):
        """
        获取模型参数，全部转为numpy数组
        """
        return {k: v.cpu().numpy() for k, v in self.state_dict().items()}

    def set_parameters(self, parameters):
        """
        用numpy数组字典设置模型参数
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in parameters:
                    param.data = torch.from_numpy(parameters[name].copy()).to(param.data.device).type(param.data.dtype)