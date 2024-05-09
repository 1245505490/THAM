import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.utils import seq_mask, masked_softmax
from models.transformer import *


class MyLinear(nn.Module):
    def __init__(self, output_dim):
        super(MyLinear, self).__init__()
        self.output_dim = output_dim
        self.linear = None

    def forward(self, x):
        if self.linear is None:
            self.linear = nn.Linear(x.size(-1), self.output_dim, device=x.device)
        output = self.linear(x)
        return output


class HierarchicalEmbedding(nn.Module):
    def __init__(self, code_levels, code_num_in_levels, code_dims):
        super(HierarchicalEmbedding, self).__init__()
        self.level_num = len(code_num_in_levels)
        self.code_levels = code_levels  # (leaf code num * level_num)
        self.level_embeddings = nn.ModuleList([nn.Embedding(code_num, code_dim)
                                               for code_num, code_dim in zip(code_num_in_levels, code_dims)])
        for emb in self.level_embeddings:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self):
        embeddings = [
            self.level_embeddings[level](self.code_levels[:, level])
            for level in range(self.level_num)
        ]

        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings


class Embedding_a(nn.Module):
    def __init__(self, code_nums, code_dims):
        super(Embedding_a, self).__init__()
        self.embeddings = nn.Parameter(nn.init.xavier_uniform_(torch.empty(code_nums + 1, code_dims)))

    def forward(self):
        return self.embeddings


class DrugEmbedding(nn.Module):
    def __init__(self, drug_num, drug_dim):
        super().__init__()
        self.drug_embeddings = nn.Parameter(data=nn.init.xavier_uniform_(torch.empty(drug_num, drug_dim)))

    def forward(self):
        return self.drug_embeddings


class GraphConvBlock(nn.Module):
    def __init__(self, node_type, input_dim, output_dim, adj):
        super().__init__()
        self.node_type = node_type
        self.adj = adj
        self.dense = nn.Linear(input_dim, output_dim)
        self.activation = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, embedding, embedding_neighbor, code_adj=None):
        output = embedding + torch.matmul(self.adj, embedding_neighbor)
        if self.node_type == 'code':
            assert code_adj is not None
            output += torch.matmul(code_adj, embedding)
        output = self.dense(output)
        output = self.bn(output)
        output = self.activation(output)
        return output


def norm_no_nan(x):
    sum_x = torch.sum(x, dim=-1, keepdim=True)
    return torch.nan_to_num(torch.div(x, sum_x), 0)


class GraphConvolution(nn.Module):
    def __init__(self, drug_dim, code_dim,
                 drug_code_adj, code_code_adj,
                 drug_hidden_dims, code_hidden_dims, device):
        super().__init__()
        self.drug_code_adj = norm_no_nan(drug_code_adj)
        self.code_drug_adj = norm_no_nan(drug_code_adj.transpose(-1, -2))
        self.code_code_adj = code_code_adj
        self.drug_blocks = []
        self.code_blocks = []
        last = drug_dim
        for layer, dim in enumerate(drug_hidden_dims):
            self.drug_blocks.append(
                GraphConvBlock('drug', input_dim=last, output_dim=dim, adj=self.drug_code_adj).to(device))
            last = dim
        last = code_dim
        for layer, dim in enumerate(code_hidden_dims):
            self.code_blocks.append(
                GraphConvBlock('code', input_dim=last, output_dim=dim, adj=self.code_drug_adj).to(device))
            last = dim
        c2d_dims = ([drug_dim] + drug_hidden_dims)[:-1]
        d2c_dims = ([code_dim] + code_hidden_dims)[:-1]
        self.c2d_linears = [MyLinear(dim) for layer, dim in enumerate(c2d_dims)]
        self.d2c_linears = [MyLinear(dim) for layer, dim in enumerate(d2c_dims)]

    def forward(self, drug_embeddings, code_embeddings):
        weight = norm_no_nan(self.code_code_adj)
        for c2d_linear, d2c_linear, drug_block, code_block in zip(self.c2d_linears, self.d2c_linears, self.drug_blocks,
                                                                  self.code_blocks):
            code_embedding_d = c2d_linear(code_embeddings)
            drug_embeddings_new = drug_block(drug_embeddings, code_embedding_d)
            drug_embeddings_c = d2c_linear(drug_embeddings)
            code_embeddings = code_block(code_embeddings, drug_embeddings_c, weight)
            drug_embeddings = drug_embeddings_new
        drug_embeddings_c = self.d2c_linears[-1](drug_embeddings)
        code_embeddings = self.code_blocks[-1](code_embeddings, drug_embeddings_c, weight)
        return drug_embeddings, code_embeddings


class VisitEmbedding(nn.Module):
    def __init__(self, max_visit_len):
        super().__init__()
        self.max_seq_len = max_visit_len

    def forward(self, code_embeddings, visit_codes, visit_lens):
        visit_codes_embedding = F.embedding(visit_codes, code_embeddings)
        visit_codes_mask = torch.unsqueeze(visit_codes > 0, dim=-1).to(dtype=visit_codes_embedding.dtype,
                                                                       device=visit_codes.device)
        visit_codes_embedding *= visit_codes_mask
        visit_codes_num = torch.unsqueeze(
            torch.sum((visit_codes > 0).to(dtype=visit_codes_embedding.dtype), dim=-1), dim=-1)
        sum_visit_codes_embedding = torch.sum(visit_codes_embedding, dim=-2)
        visits_embeddings = torch.where(visit_codes_num != 0,
                                        sum_visit_codes_embedding / torch.clamp(visit_codes_num, min=1e-9),
                                        torch.zeros_like(visit_codes_num))
        visit_mask = seq_mask(visit_lens, self.max_seq_len).unsqueeze(-1).to(dtype=visits_embeddings.dtype)
        visits_embeddings *= visit_mask
        return visits_embeddings


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, ffn_dim=1024, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim=model_dim, ffn_dim=ffn_dim, dropout=dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention


def padding_mask(seq_q, seq_k):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
    return pad_mask


class Encoder(nn.Module):
    def __init__(self, max_visit_len, num_layers, model_dim, num_heads, ffn_dim, time_dim,
                 dropout):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(model_dim=model_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout) for _ in
            range(num_layers)
        ])
        self.selection_layer = nn.Linear(1, time_dim)
        self.time_layer = nn.Linear(time_dim, model_dim)
        self.pos_embedding = PositionalEncoding(model_dim=model_dim, max_visit_len=max_visit_len)
        self.tanh = nn.Tanh()

    def forward(self, visits_embeddings, intervals, visit_mask, visit_lens):
        v_mask = visit_mask.unsqueeze(-1)
        intervals = intervals.unsqueeze(-1) / 180
        time_feature = 1 - self.tanh(torch.pow(self.selection_layer(intervals), 2))
        time_feature = self.time_layer(time_feature)
        output = visits_embeddings + time_feature
        output_pos, ind_pos = self.pos_embedding(visit_lens.unsqueeze(-1))
        output += output_pos
        output *= v_mask
        att_mask = padding_mask(ind_pos, ind_pos)
        for encoder in self.encoder_layers:
            output, _ = encoder(output, att_mask)
        return output


class Attention(nn.Module):
    def __init__(self, input_size, attention_dim):
        super().__init__()
        self.attention_dim = attention_dim
        self.u_omega = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(attention_dim, 1))))
        self.w_omega = nn.Parameter(nn.init.xavier_uniform_(torch.empty(size=(input_size, attention_dim))))

    def forward(self, x, mask=None):
        t = torch.matmul(x, self.w_omega)
        vu = torch.tensordot(t, self.u_omega, dims=1).squeeze()
        if mask is not None:
            vu *= mask
            alphas = masked_softmax(vu, mask)
        else:
            alphas = F.softmax(vu)  # (batch_size, max_seq_len)
        output = torch.sum(x * alphas.unsqueeze(-1), dim=-2)  # (batch_size, code_dim )
        return output, alphas


class TimeEncoder(nn.Module):
    def __init__(self, time_dim, quiry_dim):
        super(TimeEncoder, self).__init__()
        self.selection_layer = nn.Linear(1, time_dim)
        self.weight_layer = nn.Linear(time_dim, quiry_dim)
        self.quiry_dim = quiry_dim
        self.tanh = nn.Tanh()
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, intervals, final_queries, mask_mult):
        intervals = torch.unsqueeze(intervals, dim=2) / 180
        selection_features = 1 - self.tanh(torch.pow(self.selection_layer(intervals), 2))
        selection_features = self.leakyRelu(self.weight_layer(selection_features))
        r = torch.bmm(final_queries, selection_features.transpose(-1, -2)).squeeze()
        selection_features = r / math.sqrt(self.quiry_dim)
        selection_features = selection_features.masked_fill_(mask_mult, -torch.inf)
        return F.softmax(selection_features, dim=1)
