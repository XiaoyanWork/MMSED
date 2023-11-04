import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Function


class NonParametricClassifierOP(Function):
    
    @staticmethod
    def forward(ctx, x, y, memory, params):

        tau = params[0].item()
        # x = x.squeeze(axis=1)
        out = x.mm(memory.t())
        out.div_(tau)
        ctx.save_for_backward(x, memory, y, params)
        return out


    @staticmethod
    def backward(ctx, grad_output):
        x, memory, y, params = ctx.saved_tensors
        tau = params[0]
        momentum = params[1]

        grad_output.div_(tau)

        grad_input = grad_output.mm(memory)
        grad_input.resize_as_(x)
        
        y = y.to(x.device)

        weight_pos = memory.index_select(0, y.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(x.mul(1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return grad_input, None, None, None, None


class NonParametricClassifier(nn.Module):
    def __init__(self, args):
        super(NonParametricClassifier, self).__init__()
        
        self.args = args
        self.input_dim = self.args.npc_in_dim
        self.output_dim = self.args.train_len
        self.tau = self.args.npc_tau
        self.momentum = self.args.momentum
        
        self.register_buffer('params', torch.tensor([self.tau, self.momentum]))
        stdv = 1. / np.sqrt(self.input_dim / 3.)
        self.register_buffer(
            'memory',
            torch.rand(self.output_dim, self.input_dim).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        
        out = NonParametricClassifierOP.apply(x, y, self.memory, self.params)
        
        return out
    

class npc_Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tau = self.args.npc_loss_tau

    def forward(self, x, ff, y):

        L_id = F.cross_entropy(x, y)

        norm_ff = ff / (ff**2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_ff.t(), norm_ff)
        coef_mat.div_(self.tau)
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)
        L_fd = F.cross_entropy(coef_mat, a)
        return L_id, L_fd
    
    
class Normalize(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.power = self.args.norm_power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out