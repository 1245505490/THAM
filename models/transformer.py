import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import final_mask


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.3):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        # batch_size * num_heads, max_seq_len, max_seq_len
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        # softmax(q*k^T/根号s)*v
        # batch_size * num_heads, max_seq_len, dim_per_head
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, dropout=0.3):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        residual = query
        # key ,value,query shape:batchsize,max_seq_len, model_dim
        # attn_mask shape: batchsize,max_seq_len,max_seq_len
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        # batch_size * num_heads,max_seq_len,dim_per_head
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            # num_heads：重复次数，对应多头注意力机制的头数。
            # 1, 1： 在第二个和第三个维度上各复制一次，保持原有形状。
            # batch_size*num_heads,max_seq_len,max_seq_len
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        # 求根号下的维度，用于缩放点积注意力
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=256, ffn_dim=1024, dropout=0.3):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        # x:batch_size, max_seq_len, model_dim
        output = x
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output)

        # add residual and norm layer
        # 残差连接和层归一化
        output = self.layer_norm(x + output)
        # batch_size, max_seq_len, model_dim
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, max_visit_len):

        super(PositionalEncoding, self).__init__()
        self.max_visit_len = max_visit_len
        # 该矩阵的行数等于 max_seq_len，列数等于 modeldim。
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / model_dim) for j in range(model_dim)]
            for pos in range(max_visit_len)])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))
        # 填充行
        pad_row = torch.zeros([1, model_dim])
        position_encoding = torch.cat((pad_row, position_encoding))
        self.position_encoding = nn.Embedding(max_visit_len + 1, model_dim)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        input_pos = torch.zeros((len(input_len), self.max_visit_len), dtype=torch.int, device=input_len.device)
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                input_pos[ind, pos_ind - 1] = pos_ind
        # batchsize,max_visit_len
        # batchsize, max_visit_len, model_dim
        return self.position_encoding(input_pos), input_pos


class SingleHeadAttentionLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, attention_size):
        super().__init__()
        self.attention_size = attention_size
        self.dense_q = nn.Linear(query_size, attention_size)
        self.dense_k = nn.Linear(key_size, attention_size)
        self.dense_v = nn.Linear(query_size, value_size)

    def forward(self, q, k, v):
        # q shape: len(m2_index)+len(m3_index),graph_size
        # query shape:  len(m2_index)+len(m3_index),attentionsize
        query = self.dense_q(q)
        key = self.dense_k(k)
        value = self.dense_v(v)
        g = torch.div(torch.matmul(query, key.T), math.sqrt(self.attention_size))
        score = torch.softmax(g, dim=-1)
        # 求和的维度是num_heads,求和是将不同注意力头的输出进行融合，得到最终的注意力输出
        output = torch.sum(torch.unsqueeze(score, dim=-1) * value, dim=-2)
        return output