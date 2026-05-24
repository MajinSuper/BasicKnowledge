import torch
import matplotlib.pyplot as plt

def show_attention(attention_npy,title = 'Attention Heatmap'):
    plt.imshow(attention_npy)
    plt.colorbar()
    plt.xlabel('Key token')
    plt.ylabel('Query token')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    scores = torch.randn(6, 6)
    attention = torch.softmax(scores, dim=-1)
    show_attention(attention.detach().numpy())