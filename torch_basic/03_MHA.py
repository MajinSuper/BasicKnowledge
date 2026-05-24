import torch
from torch import nn
from visable.attn_score_visable import show_attention

class MHA(nn.Module):
    def __init__(self, emb_dim, head_num):
        super(MHA, self).__init__()
        self.emb_dim = emb_dim
        self.head_num = head_num
        assert emb_dim % head_num == 0, "emb_dim  % head_num must be 0"
        self.d_k = emb_dim // head_num

        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_o = nn.Linear(emb_dim, emb_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        # x: [B , seq, emb_dim]
        B, seq, emb_dim = x.shape
        Q = self.W_q(x)  # [B , seq, emb_dim]
        K = self.W_k(x)  # [B , seq, emb_dim]
        V = self.W_v(x)  # [B , seq, emb_dim]

        Q = Q.view(B, seq, self.head_num, self.d_k).transpose(2, 1)  # [B ,head, seq, dk]
        K = K.view(B, seq, self.head_num, self.d_k).transpose(2, 1)  # [B ,head, seq, dk]
        V = V.view(B, seq, self.head_num, self.d_k).transpose(2, 1)  # [B ,head, seq, dk]

        scores = Q @ K.transpose(-1, -2)  # [B ,head, seq, seq]
        scores = scores / (self.d_k ** 0.5)  # !! scale !!!

        if mask is not None:
            scores = scores.masked_fill(mask == 1, 1e-9)

        scores = self.softmax(scores)
        output = scores @ V  # [B ,head, seq, dk]
        output1 = output.transpose(2, 1).reshape(B, seq, self.emb_dim)  # [B, seq, emb_dim]
        # 等价于
        output2 = output.transpose(2, 1).contiguous().view(B, seq, self.emb_dim)
        print("equal:", torch.equal(output1, output2))
        return self.W_o(output1),scores


if __name__ == '__main__':
    L = 10
    emb_dim = 64
    head_num = 4
    layer = MHA(emb_dim, head_num=head_num)

    x_input = torch.rand([32, L, emb_dim])  # [B, seq, emb_dim]
    res,scores = layer(x_input)
    print(res.shape)                        # [32,100,64]
    show_attention(scores[0][0].detach().numpy(),title="Batch 0, head 0")

    causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
    show_attention(causal_mask,title="mask matric")

    res,scores = layer(x_input,causal_mask)
    print(res.shape)
    # show_attention(scores.detach().numpy())
    show_attention(scores[0][0].detach().numpy(), title="Batch 0, head 0")