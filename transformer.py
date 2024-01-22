import math

import torch
import torch.nn as nn

from utils import clones


class LayerNorm(nn.Module):
    def __init__(self, x, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(x))
        self.b = nn.Parameter(torch.zeros(x))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x-mean)/(std+self.eps) + self.b


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.shape)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """Implements x + dropout(sublayer(norm(x))) i.e., applies dropout
    residual connection to the sublayer output.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sublayer, x):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder layer has two sub-layers; self-attention &
    feed forward(point-wise)"""
    def __init__(self, size, dropout, self_attn, feed_forward):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ffn = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayers[0](lambda x: self.self_attn(x, x, x, mask), x)
        return self.sublayers[1](self.ffn, x)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layer = clones(layer, N)
        self.norm = LayerNorm(layer.shape)

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, dropout, self_attn, src_attn, ffn):
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.ffn = ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, self_mask, src_mask, tgt_mask):
        """m is the encoder output which acts as key/value and
        query 'x' comes from decoder input."""
        m = memory
        x = self.sublayer[0](lambda x: self.self_attn(x, x, x, tgt_mask), x)
        x = self.sublayer[1](lambda x: self.src_attn(x, m, m, src_mask), x)
        return self.sublayer[2](self.ffn, x)


def attention(q, k, v, mask=None, dropout=None):
    """Computes scaled dot product attention"""
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, v), p_attn


class MultiHeadAttn(nn.Module):
    """Multi head attention block"""

    def __init__(self, d_model, n_head, dropout):
        super(MultiHeadAttn, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = self.d_model // self.n_head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        # TODO: unsqueeze() the mask tensor
        nbatches = q.size(0)
        q, k, v = [
            lin(x).view(nbatches,-1,self.n_head,self.d_k).transpose(1,2)
            for lin, x in  zip(self.linears, (q,k,v))
        ]
        x, _ = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = (
            x.transpose(1,2)
            .contiguous()     # .contiguous() is called as view() requires contiguous tensor & transpose() returns non-contiguous
            .view(nbatches, -1, self.n_head*self.d_k)
        )
        del q
        del k
        del v
        return self.linears[-1](x)


class  PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(p=dropout)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(self.dropout(nn.ReLU(self.w1(x))))


# d_k = 64
# n = 10
# t = torch.rand((n,d_k))
# attn, p_attn = attention(t,t,t)
# torch.arange(10)
# torch.matmul(t, t.transpose(-1,-2))
# t = torch.arange(4)
# print(t.size())
# print(t.unsqueeze(0).size())
# print(t.unsqueeze(1).size())