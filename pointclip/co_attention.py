from math import sqrt
import torch
import torch.nn as nn
from torchsummary import summary

class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, attn_drop=0., proj_drop=0.):
        super(Self_Attention, self).__init__()
        self.qkv = nn.Linear(input_dim, input_dim * 3)
        self._norm_fact = 1 / sqrt(input_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn = None

    def forward(self, x1, x2):
        # Q = self.q(x)  # Q: batch_size * seq_len * dim_k
        # K = self.k(x)  # K: batch_size * seq_len * dim_k
        # V = self.v(x)  # V: batch_size * seq_len * dim_v
        B, C = x1.shape


        qkv1 = self.qkv(x1).reshape(B, 3, -1).permute(1, 0, 2)
            # .permute(2, 0, 3, 1)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]

        qkv2 = self.qkv(x2).reshape(B, 3, -1).permute(1, 0, 2)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]

        attn1 = (q1 @ k1.transpose(-2, -1))
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn12 = (q1 @ k2.transpose(-2, -1))
        attn21 = (q2 @ k1.transpose(-2, -1))

        attn1 = attn1.softmax(dim=-1) * self._norm_fact
        attn2 = attn2.softmax(dim=-1) * self._norm_fact
        attn12 = attn12.softmax(dim=-1) * self._norm_fact
        attn21 = attn21.softmax(dim=-1) * self._norm_fact


        attn1 = self.attn_drop(attn1)
        attn2 = self.attn_drop(attn2)
        attn12 = self.attn_drop(attn12)
        attn21 = self.attn_drop(attn21)

        x1_attn = attn1 @ v1
        x2_attn = attn2 @ v2
        x12_attn = attn12 @ v2
        x21_attn = attn21 @ v1


        x1_proj = self.proj(x1_attn + x1)
        x2_proj = self.proj(x2_attn + x2)
        x12_proj = self.proj(x12_attn + x1)
        x21_proj = self.proj(x21_attn + x2)

        #out1 = x1
        out1 = x1 + x1_attn + x12_attn + x1_proj + x12_proj
        out2 = x2 + x2_attn + x21_attn + x2_proj + x21_proj
        #out2 = x2


        return out1, out2

if __name__ == '__main__':
    batch_size = 2  # 批量数
    dim_input = 1024  # 点维度
    seq_len = 1024  # 点数量
    x1 = torch.randn(batch_size, dim_input)
    x2 = torch.randn(batch_size, dim_input)
    # xs2 = torch.randn(batch_size, dim_input)
    self_attention = Self_Attention(dim_input)
    summary(self_attention, input_size=[(1024,), (1024,)], device='cpu')
    print(x1.shape)
    print('=' * 50)
    out1, out2 = self_attention(x1, x2)
    print('=' * 50)
    print(out1.shape)