import torch


class MyLayerNorm(torch.nn.Module):
    def __init__(self, norm_shape, eps=1e-5):
        super(MyLayerNorm, self).__init__()
        self.norm_shape = norm_shape
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(norm_shape))
        self.beta = torch.nn.Parameter(torch.zeros(norm_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normed_res = (x - mean) / torch.sqrt(var + self.eps)
        return normed_res * self.gamma + self.beta


def _eval_2d_LN():
    batch_size = 2
    feature_num = 3
    in_tensor_2d = torch.rand([batch_size, feature_num])
    my_LN_layer = MyLayerNorm(feature_num)
    torch_LN_layer = torch.nn.LayerNorm(feature_num)
    my_LN_out = my_LN_layer(in_tensor_2d)
    torch_LN_out = torch_LN_layer(in_tensor_2d)
    print(f"2D tensor : {torch.allclose(my_LN_out, torch_LN_out, atol=1e-5)}")


def _eval_3d_LN():
    batch_size = 2
    feature_num = 3
    length = 4

    in_tensor_3d = torch.rand([batch_size, length, feature_num])
    my_LN_layer = MyLayerNorm(feature_num)
    torch_LN_layer = torch.nn.LayerNorm(feature_num)
    my_LN_out = my_LN_layer(in_tensor_3d)
    torch_LN_out = torch_LN_layer(in_tensor_3d)
    print(f"3D tensor : {torch.allclose(my_LN_out, torch_LN_out, atol=1e-5)}")


if __name__ == '__main__':
    _eval_2d_LN()
    _eval_3d_LN()
