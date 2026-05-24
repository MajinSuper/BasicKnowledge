import torch
import torch.nn as nn

class BatchNorm1dCustom(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if x.dim() not in (2, 3):
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D input)")

        if self.training:
            if x.dim() == 2:
                axes = (0,)
            else:
                axes = (0, 2)

            mean = x.mean(dim=axes, keepdim=True)
            var = x.var(dim=axes, unbiased=False, keepdim=True)

            momentum = self.momentum
            if momentum is None:
                momentum = 1.0 / (self.num_batches_tracked + 1).float()

            with torch.no_grad():
                self.running_mean = (1 - momentum) * self.running_mean + momentum * mean.squeeze()
                self.running_var = (1 - momentum) * self.running_var + momentum * var.squeeze()
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean.reshape(1, -1, *((1,) * (x.dim() - 2)))
            var = self.running_var.reshape(1, -1, *((1,) * (x.dim() - 2)))

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            gamma = self.gamma.reshape(1, -1, *((1,) * (x.dim() - 2)))
            beta = self.beta.reshape(1, -1, *((1,) * (x.dim() - 2)))
            x_hat = gamma * x_hat + beta

        return x_hat

# 二维输入测试
bn_official = nn.BatchNorm1d(3)
bn_custom = BatchNorm1dCustom(3)

# 同步参数
bn_custom.gamma.data = bn_official.weight.data.clone()
bn_custom.beta.data = bn_official.bias.data.clone()
bn_custom.running_mean = bn_official.running_mean.clone()
bn_custom.running_var = bn_official.running_var.clone()

x_2d = torch.randn(2, 3)
y_official = bn_official(x_2d)
y_custom = bn_custom(x_2d)
print(torch.allclose(y_official, y_custom, atol=1e-6))  # 输出: True

# 三维输入测试
x_3d = torch.randn(2, 3, 5)
bn_official_3d = nn.BatchNorm1d(3)
bn_custom_3d = BatchNorm1dCustom(3)

bn_custom_3d.gamma.data = bn_official_3d.weight.data.clone()
bn_custom_3d.beta.data = bn_official_3d.bias.data.clone()
bn_custom_3d.running_mean = bn_official_3d.running_mean.clone()
bn_custom_3d.running_var = bn_official_3d.running_var.clone()

y_official_3d = bn_official_3d(x_3d)
y_custom_3d = bn_custom_3d(x_3d)
print(torch.allclose(y_official_3d, y_custom_3d, atol=1e-6))  # 输出: True