import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.W_q = nn.Linear(emb_dim, emb_dim)  # [emb_dim , emb_dim]
        self.W_k = nn.Linear(emb_dim, emb_dim)  # [emb_dim , emb_dim]
        self.W_v = nn.Linear(emb_dim, emb_dim)  # [emb_dim , emb_dim]
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x : [B , seq , emb_dim]
        B, seq, emb_dim = x.shape
        Q = self.W_q(x)  # [B , seq , emb_dim]
        K = self.W_k(x)  # [B , seq , emb_dim]
        V = self.W_v(x)  # [B , seq , emb_dim]

        # scores = Q @ K.transpose(1, 2)  # [B,seq,dim] x [B,dim,seq] => [B,seq,seq]
        scores = torch.matmul(Q , K.transpose(1, 2))  # [B,seq,dim] x [B,dim,seq] => [B,seq,seq]
        scores = scores / (emb_dim ** 0.5)  # scale !!!
        scores = self.softmax(scores)

        output = scores @ V  # [B,seq,seq] x [B,seq,emb_dim] => [B,seq,emb_dim]
        return output

if __name__ == '__main__':
    layer = SelfAttention(128)
    input_x = torch.rand([64,100,128])

    res = layer(input_x)
    print(res.shape)