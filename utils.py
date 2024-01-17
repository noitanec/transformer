import copy

import torch
import torch.nn as nn


def subsequent_mask(size):
    attn_shape = (1,size,size)
    return (torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8) == 0).squeeze()

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
