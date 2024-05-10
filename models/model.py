import numpy as np
import torch
from torch import nn
from models.layers import *
from models.utils import seq_mask, final_mask, get_multmask


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        output = self.dropout(x)
        output = self.linear(output)
        return output


class Model(nn.Module):
    def __init__(self, config, hyper_params):
        super().__init__()
        device = config['device']
        self.cgl = HGLFeatureExtractor(config, hyper_params).to(device)
        self.classifier = Classifier(input_size=hyper_params['input_dim'], output_size=hyper_params['output_dim'],
                                     dropout_rate=hyper_params['dropout']).to(device)

    def forward(self, visit_codes, visit_lens, intervals):
        inputs = {
            'visit_codes': visit_codes,
            'visit_lens': visit_lens,
            'intervals': intervals
        }
        output = self.cgl(inputs)
        output = self.classifier(output)
        return output


class HGLFeatureExtractor(nn.Module):
    def __init__(self, config, hyper_params):
        super().__init__()
        self.config = config
        self.hyper_params = hyper_params
        self.device = config['device']
        self.hierarchical_embedding_layer = HierarchicalEmbedding(code_levels=config['code_levels'],
                                                                  code_num_in_levels=config['code_num_in_levels'],
                                                                  code_dims=hyper_params['code_dims']).to(self.device)
        self.drug_embedding_layer = DrugEmbedding(drug_num=config['drug_num'], drug_dim=hyper_params['drug_dim']).to(
            self.device)
        code_dim = np.sum(hyper_params['code_dims'])
        drug_dim = hyper_params['drug_dim']
        self.max_visit_len = config['max_visit_seq_len']
        self.gcn = GraphConvolution(drug_dim=drug_dim, code_dim=code_dim, drug_code_adj=config['drug_code_adj'],
                                    code_code_adj=config['code_code_adj'],
                                    drug_hidden_dims=hyper_params['drug_hidden_dims'],
                                    code_hidden_dims=hyper_params['code_hidden_dims'], device=self.device).to(
            self.device)
        self.visit_embedding_layer = VisitEmbedding(max_visit_len=self.max_visit_len).to(self.device)
        self.feature_encoder = Encoder(max_visit_len=config['max_visit_seq_len'],
                                       num_layers=hyper_params['num_layers'], model_dim=code_dim,
                                       num_heads=hyper_params['num_heads'], ffn_dim=hyper_params['ffn_dim'],
                                       time_dim=hyper_params['time_dim'],
                                       dropout=hyper_params['dropout']).to(self.device)
        self.quiry_layer = nn.Linear(code_dim, hyper_params['quiry_dim'])
        self.time_encoder = TimeEncoder(time_dim=hyper_params['time_dim'], quiry_dim=hyper_params['quiry_dim']).to(
            self.device)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.quiry_weight_layer = nn.Linear(code_dim, 2)
        self.quiry_weight_layer2 = nn.Linear(code_dim, 1)
        self.attention = Attention(code_dim, attention_dim=hyper_params['attention_dim']).to(self.device)

    def forward(self, inputs):
        visit_codes = inputs['visit_codes']
        visit_lens = inputs['visit_lens']
        intervals = inputs['intervals']
        visit_mask = seq_mask(visit_lens, self.max_visit_len)
        mask_final = final_mask(visit_lens, self.max_visit_len)
        mask_mult = get_multmask(visit_mask)
        code_embeddings = self.hierarchical_embedding_layer()
        drug_embeddings = self.drug_embedding_layer()
        drug_embeddings, code_embeddings = self.gcn(drug_embeddings=drug_embeddings, code_embeddings=code_embeddings)
        visits_embeddings = self.visit_embedding_layer(code_embeddings=code_embeddings, visit_codes=visit_codes,
                                                       visit_lens=visit_lens)
        features = self.feature_encoder(visits_embeddings, intervals, visit_mask, visit_lens)
        final_statues = features * mask_final.unsqueeze(-1)
        final_statues = final_statues.sum(1, keepdim=True)
        quiryes = self.leakyRelu(self.quiry_layer(final_statues))
        _, self_weight = self.attention(features, visit_mask)
        self_weight = self_weight.unsqueeze(-1)
        time_weight = self.time_encoder(intervals, quiryes, mask_mult).unsqueeze(-1)
        attention_weight = torch.softmax(self.quiry_weight_layer2(final_statues), 2)
        total_weight = torch.cat((time_weight, self_weight), 2)
        total_weight = torch.sum(total_weight * attention_weight, 2, keepdim=True)
        total_weight = total_weight / (torch.sum(total_weight, 1, keepdim=True) + 1e-8)
        weighted_features = features * total_weight
        output = torch.sum(weighted_features, 1)
        return output
