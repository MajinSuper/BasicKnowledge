import torch
from torch import nn


class MQA(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int):
        super(MQA, self).__init__()
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        assert (self.hidden_dim % head_num == 0), "hidden_dim must be divisible by head_num"
        self.dk = self.hidden_dim // self.head_num

        self.W_q = nn.Linear(self.hidden_dim, self.head_num * self.dk)
        self.W_k = nn.Linear(self.hidden_dim,  self.dk)
        self.W_v = nn.Linear(self.hidden_dim,  self.dk)
        self.W_o = nn.Linear(self.head_num * self.dk , self.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.k_cache = []
        self.v_cache = []

    def reset_cache(self):
        self.k_cache = []
        self.v_cache = []

    def forward(self, x: torch.Tensor, mask=None):
        # x : [B, seq, dim]
        B, seq, dim = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        # ！！ 切分多头 ！！
        Q = Q.view(B, seq, self.head_num, self.dk).transpose(2, 1)  # [B, head_num, seq, dk]
        K = K.view(B, seq, 1, self.dk).transpose(2, 1)  # [B, 1, seq, dk]
        V = V.view(B, seq, 1, self.dk).transpose(2, 1)  # [B, 1, seq, dk]

        # ！！计算注意力 ！！
        K = K.repeat_interleave(self.head_num, dim=1)            # [B, head_num, seq, dk]
        V = V.repeat_interleave(self.head_num, dim=1)            # [B, head_num, seq, dk]
        attn_score = Q @ K.transpose(-2, -1) / (self.dk ** 0.5)  # [B, head_num, seq, seq]
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 1, -1e9)
        attn_score = self.softmax(attn_score)
        res = attn_score @ V  # [B, head_num, seq, seq] x [B, head_num, seq, dk]  => [B, head_num, seq, dk]

        # ！！ 合并多头 ！！
        return self.W_o(res.transpose(2, 1).reshape(B, seq, -1))


if __name__ == '__main__':
    B, seq, hidden_dim = 32, 100, 64
    cacual_mask = torch.triu(torch.ones(seq, seq), diagonal=1)
    layer = MQA(hidden_dim=hidden_dim, head_num=4)
    x = torch.rand([B, seq, hidden_dim])
    y1 = layer(x, cacual_mask)  # [B,seq,hidden_dim]
    print(y1.shape)
