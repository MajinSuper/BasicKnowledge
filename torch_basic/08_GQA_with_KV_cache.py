import torch
from torch import nn


class MQA(nn.Module):
    def __init__(self, hidden_dim: int, head_num: int,kv_head_num: int):
        super(MQA, self).__init__()
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.kv_head_num = kv_head_num
        assert (self.hidden_dim % head_num == 0), "hidden_dim must be divisible by head_num"
        self.dk = self.hidden_dim // self.head_num
        assert self.head_num % self.kv_head_num == 0, "dk must be divisible by head_num"

        self.W_q = nn.Linear(self.hidden_dim, self.head_num * self.dk)
        self.W_k = nn.Linear(self.hidden_dim, self.kv_head_num * self.dk)
        self.W_v = nn.Linear(self.hidden_dim, self.kv_head_num * self.dk)
        self.W_o = nn.Linear(self.head_num * self.dk , self.hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.k_cache = []
        self.v_cache = []

    def reset_cache(self):
        self.k_cache = []
        self.v_cache = []

    def forward(self, x: torch.Tensor, mask=None, use_kvcahce=False):
        # x : [B, seq_in, dim]
        # seq_in = 1 when kv cache attention
        # seq_in = seq when full attention
        B, seq, dim = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)

        # ！！ 切分多头 ！！
        Q = Q.view(B, seq, self.head_num, self.dk).transpose(2, 1)  # [B, head_num, seq_in, dk]
        K = K.view(B, seq, self.kv_head_num, self.dk).transpose(2, 1)  # [B, kv_head_num, seq_in, dk]
        V = V.view(B, seq, self.kv_head_num, self.dk).transpose(2, 1)  # [B, kv_head_num, seq_in, dk]

        if use_kvcahce:
            self.k_cache.append(K)  # [ [B, kv_head_num, dk] ... ]
            self.v_cache.append(V)  # [ [B, kv_head_num, dk] ... ]
            K = torch.cat(self.k_cache, dim=2)  # [B, kv_head_num, seq, dk]
            V = torch.cat(self.v_cache, dim=2)  # [B, kv_head_num, seq, dk]

        # ！！计算注意力 ！！
        K = K.repeat_interleave(self.head_num // self.kv_head_num, dim=1)   # [B, head_num, seq, dk]
        V = V.repeat_interleave(self.head_num // self.kv_head_num, dim=1)   # [B, head_num, seq, dk]
        attn_score = Q @ K.transpose(-2, -1) / (self.dk ** 0.5)             # [B, head_num, seq_in, seq]
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 1, -1e9)
        attn_score = self.softmax(attn_score)
        res = attn_score @ V  # [B, head_num, seq_in, seq] x [B, head_num, seq, dk]  => [B, head_num, seq_in, dk]

        # ！！ 合并多头 ！！
        return self.W_o(res.transpose(2, 1).reshape(B, seq, -1))


if __name__ == '__main__':
    B, seq, hidden_dim = 32, 100, 64
    cacual_mask = torch.triu(torch.ones(seq, seq), diagonal=1)
    layer = MQA(hidden_dim=hidden_dim, head_num=8,kv_head_num=4)
    x = torch.rand([B, seq, hidden_dim])
    y1 = layer(x, cacual_mask)  # [B,seq,hidden_dim]

    y2 = []
    for xid in range(seq):
        x_cur = x[:, xid, :].unsqueeze(1)
        y_cur = layer(x_cur, use_kvcahce=True)
        y2.append(y_cur)
    y2 = torch.cat(y2, dim=1)  # [B,seq,hidden_dim]

    print(y1.shape)
    print(y2.shape)
    print(torch.allclose(
        y1,
        y2,
        atol=1e-6
    ))
    print(y1)
    print(y2)
