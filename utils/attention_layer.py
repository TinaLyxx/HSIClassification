import torch
from torch import nn
import torch.nn.functional as F



class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"


        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_linear(attn_output)

        return output


class Spa_Attention(CrossAttention):
    def forward(self, query, key, value):
        batch_size = query.size(0)

        Q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)


        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.sigmoid(attn_scores)

        attn_output = V * attn_weights.expand(-1,-1,self.head_dim,-1).permute(0,1,3,2)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

        output = self.out_linear(attn_output) # B,L,C
        return output


class Spe_Attention(nn.Module):
    def __init__(self, embed_dim, seq_len):
        super(Spe_Attention, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.maxpool = nn.MaxPool1d(kernel_size=seq_len)
        self.avgpool = nn.AvgPool1d(kernel_size=seq_len)
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=2, stride=1, padding=0, groups=embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias = False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_tran = x.permute(0, 2, 1) # B,C,L
        x_cat = torch.cat([self.maxpool(x_tran), self.avgpool(x_tran)], dim=2) # B,C,2
        x_cat = self.conv1d(x_cat).permute(0, 2, 1) # B,1,C
        q = self.proj(x_cat).expand(-1, self.seq_len, -1) # B,L,C

        output = x * q

        return output

