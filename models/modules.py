import torch
import torch.nn as nn
import torch.nn.functional as F
from torchfm.layer import FactorizationMachine, FeaturesLinear, MultiLayerPerceptron
import numpy as np

from models.pep_embedding import PEPEmbedding


class LR(torch.nn.Module):
    def __init__(self, opt):
        super(LR, self).__init__()
        self.use_cuda = opt.get('use_cuda')
        self.field_dims = opt['field_dims']
        self.linear = FeaturesLinear(self.field_dims)  # linear part

    def forward(self, x):
        """Compute Score"""
        score = self.linear.forward(x)
        return score.squeeze(1)

    def l2_penalty(self, x, lamb):
        return 0

    def calc_sparsity(self):
        return 0, 0

    def get_threshold(self):
        return 0

    def get_embedding(self):
        return np.zeros(1)


class FM(torch.nn.Module):
    """Factorization Machines"""

    def __init__(self, opt):
        super(FM, self).__init__()
        self.use_cuda = opt.get('use_cuda')
        self.latent_dim = opt['latent_dim']
        self.field_dims = opt['field_dims']

        self.feature_num = sum(self.field_dims)
        self.embedding = PEPEmbedding(opt)
        self.linear = FeaturesLinear(self.field_dims)  # linear part
        self.fm = FactorizationMachine(reduce_sum=True)
        print("BackBone Embedding Parameters: ", self.feature_num * self.latent_dim)

    def forward(self, x):
        linear_score = self.linear.forward(x)
        xv = self.embedding(x)
        fm_score = self.fm.forward(xv)
        score = linear_score + fm_score
        return score.squeeze(1)

    def l2_penalty(self, x, lamb):
        xv = self.embedding(x)
        xv_sq = xv.pow(2)
        xv_penalty = xv_sq * lamb
        xv_penalty = xv_penalty.sum()
        return xv_penalty

    def calc_sparsity(self):
        base = self.feature_num * self.latent_dim
        non_zero_values = torch.nonzero(self.embedding.sparse_v).size(0)
        percentage = 1 - (non_zero_values / base)
        return percentage, non_zero_values

    def get_threshold(self):
        return self.embedding.g(self.embedding.s)

    def get_embedding(self):
        return self.embedding.sparse_v.detach().cpu().numpy()


class DeepFM(FM):
    def __init__(self, opt):
        super(DeepFM, self).__init__(opt)
        self.embed_output_dim = len(self.field_dims) * self.latent_dim
        self.mlp_dims = opt['mlp_dims']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=0.2)

    def forward(self, x):
        linear_score = self.linear.forward(x)
        xv = self.embedding(x)
        fm_score = self.fm.forward(xv)
        dnn_score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        score = linear_score + fm_score + dnn_score
        return score.squeeze(1)


class AutoInt(DeepFM):
    def __init__(self, opt):
        super(AutoInt, self).__init__(opt)
        self.has_residual = opt['has_residual']
        self.full_part = opt['full_part']
        self.atten_embed_dim = opt['atten_embed_dim']
        self.num_heads = opt['num_heads']
        self.num_layers = opt['num_layers']
        self.att_dropout = opt['att_dropout']

        self.atten_output_dim = len(self.field_dims) * self.atten_embed_dim
        self.dnn_input_dim = len(self.field_dims) * self.latent_dim

        self.atten_embedding = torch.nn.Linear(self.latent_dim, self.atten_embed_dim)
        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(self.atten_embed_dim, self.num_heads, dropout=self.att_dropout) for _ in range(self.num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(self.latent_dim, self.atten_embed_dim)

    def forward(self, x):
        xv = self.embedding(x)
        score = self.autoint_layer(xv)
        if self.full_part:
            dnn_score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
            score = dnn_score + score

        return score.squeeze(1)

    def autoint_layer(self, xv):
        """Multi-head self-attention layer"""
        atten_x = self.atten_embedding(xv)  # bs, field_num, atten_dim
        cross_term = atten_x.transpose(0, 1)  # field_num, bs, atten_dim
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)
        cross_term = cross_term.transpose(0, 1)  # bs, field_num, atten_dim
        if self.has_residual:
            V_res = self.V_res_embedding(xv)
            cross_term += V_res
        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)  # bs, field_num * atten_dim
        output = self.attn_fc(cross_term)
        return output




