import torch

x = torch.rand([2,3,16])

linear_layer = torch.nn.Linear(16,1024)

res = linear_layer(x)
print(res.shape)    
