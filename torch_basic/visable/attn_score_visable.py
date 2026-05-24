import torch
import matplotlib.pyplot as plt

scores = torch.randn(6, 6)

attention = torch.softmax(scores, dim=-1)