import torch
import math
from .prec_grad_maker import PreconditionedGradientMaker, PreconditioningConfig
import torch.nn as nn

class AdamGradientMaker(PreconditionedGradientMaker):
    """
    implements ADAM Algorithm, as a preceding step.
    """
    def __init__(self, model: nn.Module, config, optimizer, betas = (0.9,0.99), eps = 1e-8):
        super().__init__(model, config)
        self.optim = optimizer
        self.betas = betas
        self.eps = eps
        self.momentum = optimizer.param_groups[0]['momentum']
        optimizer.param_groups[0]['momentum'] = 0
    
    @torch.no_grad()
    def precondition(self):
        """
        Performs a single optimization step.
        """
        loss = None
        for group in self.optim.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.optim.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Momentum (Exponential MA of gradients)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    #print(p.data.size())
                    # RMS Prop componenet. (Exponential MA of squared gradients). Denominator.
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                b1, b2 = self.betas
                state['step'] += 1
                
                # L2 penalty. Gotta add to Gradient as well.
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Momentum
                exp_avg = torch.mul(exp_avg, b1) + (1 - b1)*grad
                # RMS
                exp_avg_sq = torch.mul(exp_avg_sq, b2) + (1-b2)*(grad*grad)
                
                denom = exp_avg_sq.sqrt() + self.eps

                bias_correction1 = 1 / (1 - b1 ** state['step'])
                bias_correction2 = 1 / (1 - b2 ** state['step'])
                
                adapted_learning_rate = bias_correction1 / math.sqrt(bias_correction2)

                p.grad = adapted_learning_rate * exp_avg / denom 
                
                
        return loss