# Model Utils

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    # return number of parameters
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_grad(self):
        # More efficient than optimizer.zero_grad() according to:
        # Szymon Migacz "PYTORCH PERFORMANCE TUNING GUIDE" at GTC-21.
        # - doesn't execute memset for every parameter
        # - memory is zeroed-out by the allocator in a more efficient way
        # - backward pass updates gradients with "=" operator (write) (unlike
        # zero_grad() which would result in "+=").
        # In PyT >= 1.7, one can do `model.zero_grad(set_to_none=True)`
        for p in self.parameters():
            p.grad = None

    def print_params(self):
        for p in self.named_parameters():
            print(p)


class Attention(nn.Module):
    def __init__(self, num_head, res_dim, head_dim,
                 dropout=0.0, dropatt=0.0, pre_lnorm=True):
        super().__init__()

        self.num_head = num_head
        self.res_dim = res_dim
        self.head_dim = head_dim
        self.dropout = dropout

        self.slow_net = nn.Linear(
            res_dim, num_head * 3 * head_dim, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.out_linear = nn.Linear(num_head * head_dim, res_dim, bias=False)

        self.layer_norm = nn.LayerNorm(res_dim)

        self.scale = 1 / (res_dim ** 0.5)

        self.pre_lnorm = pre_lnorm

    def forward(self, x):
        # (len, B, dim)

        slen, bsz, _ = x.shape
        x_in = self.layer_norm(x) if self.pre_lnorm else x

        qkv = self.slow_net(x_in)
        qkv = qkv.view(slen, bsz, self.num_head, 3 * self.head_dim)
        head_q, head_k, head_v = torch.split(
            qkv, (self.head_dim,) * 3, -1)

        # (qlen, klen, B, num_head)
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        attn_score = F.softmax(attn_score, dim=1)
        attn_score = self.dropatt(attn_score)

        # (qlen, klen, B, num_head), (klen, B, num_head, head_dim)
        # --> (qlen, B, num_head, head_dim)
        attn_out = torch.einsum('ijbn,jbnd->ibnd', (attn_score, head_v))
        attn_out = attn_out.contiguous().view(
            slen, bsz, self.num_head * self.head_dim)

        attn_out = self.out_linear(attn_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            output = x + attn_out
        else:
            output = self.layer_norm(x + attn_out)

        # (len, B, dim)
        return output


# A block of residual feed-forward layers in Transformer
class TransformerFFlayers(nn.Module):
    def __init__(self, ff_dim, res_dim, dropout, pre_lnorm=True):
        super(TransformerFFlayers, self).__init__()

        self.res_dim = res_dim
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.pre_lnorm = pre_lnorm

        self.ff_layers = nn.Sequential(
            nn.Linear(res_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, res_dim),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(res_dim)

    def forward(self, x):
        if self.pre_lnorm:
            return self.ff_layers(self.layer_norm(x)) + x
        else:
            return self.layer_norm(self.ff_layers(x) + x)


# Fast weight layer with feed-forward fast net,
# with recurrent update rule.
class RFWPlayer(nn.Module):
    def __init__(self, num_head, dim_head, in_dim, dropout):
        super(RFWPlayer, self).__init__()

        self.num_head = num_head
        self.dim_head = dim_head
        self.in_dim = in_dim

        from rec_update_fwm_tanh import rec_update_fwm_tanh
        self.fw_layer = rec_update_fwm_tanh

        self.slow_net = nn.Linear(
            in_dim, num_head * (3 * dim_head + 1), bias=False)

        self.R_q = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.R_k = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.R_v = nn.Parameter(torch.Tensor(1, num_head, dim_head, dim_head),
                                requires_grad=True)
        self.r_b = nn.Parameter(torch.Tensor(1, num_head, 1, dim_head),
                                requires_grad=True)

        self.layer_norm = nn.LayerNorm(in_dim)
        self.out_linear = nn.Linear(num_head * dim_head, in_dim, bias=False)

        self.drop = nn.Dropout(dropout)

        self.reset_rec_parameters()

    def reset_rec_parameters(self):
        std = 1.0 / math.sqrt(self.dim_head)
        nn.init.normal_(self.R_q, mean=0., std=std)
        nn.init.normal_(self.R_k, mean=0., std=std)
        nn.init.normal_(self.R_v, mean=0., std=std)
        nn.init.normal_(self.r_b, mean=0., std=std)

    def forward(self, x, state0=None):
        # x shape: (len, B, n_head * d_head)
        slen, bsz, _ = x.size()

        out = self.layer_norm(x)

        qkvb = self.slow_net(out)
        qkvb = qkvb.view(slen, bsz, self.num_head, 3 * self.dim_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.dim_head,) * 3 + (1,), -1)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        fast_weights = torch.zeros(
            bsz, self.num_head, self.dim_head, self.dim_head,
            device=head_k.device)

        if state0 is None:
            state0 = torch.zeros(
                bsz, self.num_head, 1, self.dim_head, device=head_k.device)

        out = self.fw_layer(head_q, head_k, head_v, head_beta,
                            self.R_q, self.R_k, self.R_v, self.r_b,
                            fast_weights, state0)

        out = out.transpose(1, 2)
        out = out.reshape(bsz, slen, self.num_head * self.dim_head)
        out = out.transpose(0, 1)
        # expect [qlen, B, n_head * d_head]

        # linear projection
        out = self.out_linear(out)
        out = self.drop(out)
        out = x + out

        return out
