import torch


class MyBatchNorm1d(torch.nn.Module):
    def __init__(self, feature_num, eps=1e-5, momentum=0.1):
        super().__init__()
        self.feature_num = feature_num  # feature numbers
        self.eps = eps  # avoid var = 0

        # moving average
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(feature_num))
        self.register_buffer("running_var", torch.ones(feature_num))

        # learnable parameters
        self.gamma = torch.nn.Parameter(torch.ones(feature_num))
        self.beta = torch.nn.Parameter(torch.zeros(feature_num))

    def forward(self, x):
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError(f"Excepted 2D or 3D tensor , but get {x.dim()}")

        if self.training:  # train
            if x.dim() == 2:
                dims = [0]
            else:
                dims = [0, 2]

            # calc mean and var
            mean_value = x.mean(dim=dims, keepdims=True)
            var_value = x.var(dim=dims, unbiased=False, keepdims=True)

            # update moving average of mean and var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_value
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_value
        else:  # eval
            if x.dim() == 2:
                mean_value = self.running_mean  # [feature_num]
                var_value = self.running_var  # [feature_num]
            else:
                mean_value = self.running_mean.unsqueeze(-1).unsqueeze(0)  # [1,feature_num,1]
                var_value = self.running_var.unsqueeze(-1).unsqueeze(0)  # [1,feature_num,1]

        normalized_value = (x - mean_value) / torch.sqrt(var_value + self.eps)

        if x.dim() == 2:
            return normalized_value * self.gamma + self.beta
        else:
            return normalized_value * self.gamma.unsqueeze(-1) + self.beta.unsqueeze(-1)


def _eval_2d_BN():
    batch_size = 2
    feature_nums = 3

    # input
    in_tensor_2d = torch.rand([batch_size, feature_nums])

    # build layer
    my_bn_layer = MyBatchNorm1d(feature_nums)
    torch_bn_layer = torch.nn.BatchNorm1d(feature_nums, momentum=0.1)

    my_bn_layer.train()
    torch_bn_layer.train()

    # forward
    my_out = my_bn_layer(in_tensor_2d)
    torch_out = torch_bn_layer(in_tensor_2d)

    # valid or not
    print(f"[train] batch norm for 2d-tensor :{torch.allclose(my_out, torch_out)}")

    my_bn_layer.eval()
    torch_bn_layer.eval()

    my_bn_layer.gamma.data = torch_bn_layer.weight.data.clone()
    my_bn_layer.beta.data = torch_bn_layer.bias.data.clone()
    my_bn_layer.running_mean = torch_bn_layer.running_mean.clone()
    my_bn_layer.running_var = torch_bn_layer.running_var.clone()

    # forward
    my_out = my_bn_layer(in_tensor_2d)
    torch_out = torch_bn_layer(in_tensor_2d)

    # valid or not
    print(f"[eval] batch norm for 2d-tensor :{torch.allclose(my_out, torch_out)}")


def _eval_3d_BN():
    global my_bn_layer
    batch_size = 2
    feature_nums = 3
    length = 4
    # input
    in_tensor_3d = torch.rand([batch_size, feature_nums, length])
    # build layer
    my_bn_layer = MyBatchNorm1d(feature_nums)
    torch_bn_layer = torch.nn.BatchNorm1d(feature_nums, momentum=0.1)
    my_bn_layer.train()
    torch_bn_layer.train()
    # forward
    my_out = my_bn_layer(in_tensor_3d)
    torch_out = torch_bn_layer(in_tensor_3d)
    # valid or not
    print(f"[train] batch norm for 3d-tensor :{torch.allclose(my_out, torch_out)}")
    my_bn_layer.eval()
    torch_bn_layer.eval()
    my_bn_layer.gamma.data = torch_bn_layer.weight.data.clone()
    my_bn_layer.beta.data = torch_bn_layer.bias.data.clone()
    my_bn_layer.running_mean = torch_bn_layer.running_mean.clone()
    my_bn_layer.running_var = torch_bn_layer.running_var.clone()
    # forward
    my_out = my_bn_layer(in_tensor_3d)
    torch_out = torch_bn_layer(in_tensor_3d)
    # valid or not
    print(f"[eval] batch norm for 3d-tensor :{torch.allclose(my_out, torch_out)}")


if __name__ == '__main__':
    _eval_2d_BN()

    _eval_3d_BN()
