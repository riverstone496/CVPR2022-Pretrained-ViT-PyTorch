import math
import warnings
from dataclasses import dataclass
from typing import List, Any, Iterable
from torch import Tensor

import torch
import torch.nn as nn
from .precondition.prec_grad_maker import PreconditionedGradientMaker
from . import GradientMaker
from .fisher import LOSS_CROSS_ENTROPY,SHAPE_LAYER_WISE,SHAPE_KRON,FISHER_MC
from .fisher import FisherConfig, get_fisher_maker
from .vector import ParamVector

import copy

__all__ = ['SensitivityMaker']

class SensitivityMaker(PreconditionedGradientMaker):
    def __init__(self, grad_maker: PreconditionedGradientMaker, use_functorch = False):
        self.grad_maker = grad_maker
        self.model=grad_maker.model
        self.use_functorch = use_functorch
        self.data_size = grad_maker.config.data_size
        if hasattr(self.grad_maker,'module_dict'):
            self.module_dict = self.grad_maker.module_dict
        else:
            self.module_dict = nn.ModuleDict({name.replace('.', '_'): m for name, m in self.model.named_modules()
                                          if self._is_supported(name, m)})
        self.module_inverse_dict = inverse_dict(self.module_dict)

    def simple_sensitivity(self,grad_maker: PreconditionedGradientMaker,sgd_grad=False):
        grad_maker.forward()
        params = list(self.module_dict.parameters())
        grads = list(torch.autograd.grad(grad_maker._loss, params))
        sensitivity_all = 0
        if sgd_grad:
            Pgrads=copy.copy(grads)
        else:
            Pgrads = [p.grad for p in self.module_dict.parameters()]
        module_sensitivity={}

        for module in self.module_dict.children():
            G = grads.pop(0)
            PG = Pgrads.pop(0)
            if isinstance(module, nn.Conv2d):
                G = G.flatten(start_dim=1)
                PG = PG.flatten(start_dim=1)
            if module.bias is not None:
                G = torch.cat([G, grads.pop(0).unsqueeze(-1)], dim=1)
                PG = torch.cat([PG, Pgrads.pop(0).unsqueeze(-1)], dim=1)

            c = parameters_to_vector(G).dot(parameters_to_vector(PG))
            sensitivity_all += c

            mname=self.module_inverse_dict[module]
            module_sensitivity[mname]=float(c)
        module_sensitivity['sensitivity_all']=float(sensitivity_all)

        assert len(Pgrads) == 0
        assert len(grads) == 0

        return module_sensitivity

    def hessian_sensitivity(self,grad_maker: PreconditionedGradientMaker,lr: float,sgd_grad=False):
        sensitivity_all=0
        grad_maker.forward()
        params = list(self.module_dict.parameters())
        grads = list(torch.autograd.grad(grad_maker._loss, params,create_graph=True))
        if sgd_grad:
            Pgrads=copy.copy(grads)
        else:
            Pgrads = [p.grad for p in self.module_dict.parameters()]
        module_sensitivity={}

        if self.use_functorch:
            Mvs = grad_maker.loss_hvp(tangents=tuple(Pgrads),accumulate_grad=False)
            Mvs=list(Mvs)
        else:
            Mvs = list(torch.autograd.grad(grads, params, Pgrads))

        for module in self.module_dict.children():
            G = grads.pop(0)
            PG = Pgrads.pop(0)
            HPG = Mvs.pop(0)

            if isinstance(module, nn.Conv2d):
                G = G.flatten(start_dim=1)
                PG = PG.flatten(start_dim=1)
                HPG = HPG.flatten(start_dim=1)
            if module.bias is not None:
                G = torch.cat([G, grads.pop(0).unsqueeze(-1)], dim=1)
                PG = torch.cat([PG, Pgrads.pop(0).unsqueeze(-1)], dim=1)
                HPG = torch.cat([HPG, Mvs.pop(0).unsqueeze(-1)], dim=1)

            c1=lr*parameters_to_vector(G).dot(parameters_to_vector(PG))
            c2=-0.5*(lr**2)*parameters_to_vector(PG).dot(parameters_to_vector(HPG))
            c=c1+c2
            sensitivity_all+=c
            mname=self.module_inverse_dict[module]
            module_sensitivity[mname]=float(c)
        module_sensitivity['sensitivity_all']=float(sensitivity_all)
        assert len(Pgrads) == 0
        assert len(Mvs) == 0
        return module_sensitivity

    def kl_sensitivity(self,grad_maker: PreconditionedGradientMaker,sgd_grad=False):
        sensitivity_all=0
        params = list(self.module_dict.parameters())
        if sgd_grad:
            grad_maker.forward()
            grads = list(torch.autograd.grad(grad_maker._loss, params,create_graph=True))
            Pgrads=(grads)
        else:
            Pgrads = [p.grad for p in params]

        module_sensitivity={}

        if self.use_functorch:
            Mvs = self.grad_maker.fvp(loss_type = LOSS_CROSS_ENTROPY,data_size = self.data_size,tangents=tuple(Pgrads))
            Mvs = list(Mvs)
        else:
            Pgrads = ParamVector(params,Pgrads)
            fisher_config = FisherConfig(fisher_type=FISHER_MC,fisher_shapes=SHAPE_LAYER_WISE,loss_type=LOSS_CROSS_ENTROPY,data_size=self.data_size,fvp_attr='fvp',ignore_modules=self.grad_maker.config.ignore_modules)
            fisher_maker = get_fisher_maker(model=self.model,config=fisher_config)
            fisher_maker.setup_model_call(self.grad_maker._model_fn,*self.grad_maker._model_args)
            fisher_maker.setup_loss_call(self.grad_maker._loss_fn, *self.grad_maker._loss_fn_args)
            fvp_fn = fisher_maker._get_fvp_fn()
            Mvs=fvp_fn(vec=Pgrads)
            Pgrads,Mvs=list(Pgrads.values()),list(Mvs.values())

        for module in self.module_dict.children():
            FPG = Mvs.pop(0)
            PG = Pgrads.pop(0)
            if isinstance(module, nn.Conv2d):
                FPG = FPG.flatten(start_dim=1)
                PG = PG.flatten(start_dim=1)
            if module.bias is not None:
                FPG = torch.cat([FPG, Mvs.pop(0).unsqueeze(-1)], dim=1)
                PG = torch.cat([PG, Pgrads.pop(0).unsqueeze(-1)], dim=1)

            c=parameters_to_vector(PG).dot(parameters_to_vector(FPG))
            sensitivity_all+=c
            mname=self.module_inverse_dict[module]
            module_sensitivity[mname]=float(c)
        module_sensitivity['sensitivity_all']=float(sensitivity_all)
        assert len(Pgrads) == 0
        assert len(Mvs) == 0
        return module_sensitivity

    def cos_norm_sim(self,grad_maker: PreconditionedGradientMaker):
        module_sensitivity={}
        module_norm_sensitivity={}
        params = list(grad_maker.module_dict.parameters())
        grad_maker.forward()
        grads = list(torch.autograd.grad(grad_maker._loss, params,create_graph=True))
        Pgrads = [p.grad for p in params]

        for module in self.module_dict.children():
            G = grads.pop(0)
            PG = Pgrads.pop(0)
            cos = nn.CosineSimilarity(dim=0)
            c=cos(parameters_to_vector(G),parameters_to_vector(PG))
            n=torch.norm(parameters_to_vector(PG))/torch.norm(parameters_to_vector(G))
            mname=self.module_inverse_dict[module]
            module_sensitivity[mname]=float(c)
            module_norm_sensitivity[mname]=float(n)
        return module_sensitivity,module_norm_sensitivity

    def _is_supported(self, module_name: str, module: nn.Module) -> bool:
        if len(list(module.children())) > 0:
            return False
        if all(not p.requires_grad for p in module.parameters()):
            return False
        return True

    def criterion(self, mat_type, n_samples=1, damping=None,
        criterion_types=[1,2,3,4], use_functorch=False):
        criterion_dic = {}
        
        if damping is None:
            damping = self.config.damping

        if mat_type != 'fisher_without_damping':
            self.add_damping()
        else:
            damping=0

        for ctype in criterion_types:
            criterion_dic['criterion'+str(ctype)]={}
            criterion_dic['SGD_criterion'+str(ctype)]={}
            criterion_dic['SGD_ratio_criterion'+str(ctype)]={}
            for module in self.module_dict.children():
                setattr(module,'criterion'+str(ctype),0)
                setattr(module,'criterion_sgd'+str(ctype),0)

        for i in range(n_samples):
            torch.cuda.manual_seed(int(torch.rand(1) * 100))
            params = list(self.module_dict.parameters())
            vs = [torch.randn_like(p) for p in params]

            if use_functorch:
                if mat_type == 'hessian':
                    Mvs = self.loss_hvp(tangents=tuple(vs),accumulate_grad=False)
                elif mat_type == 'fisher' or mat_type == 'fisher_without_damping':
                    Mvs = self.fvp(loss_type = LOSS_CROSS_ENTROPY,data_size = self.config.data_size,tangents=tuple(vs))
                    Mvs = damp_vector(Mvs,vs,damping)
            else:
                if mat_type == 'hessian':
                    self.forward()
                    grads = torch.autograd.grad(self._loss, params, create_graph=True)
                    for p, g in zip(params, grads):
                        p.prev_grad = p.grad.detach().clone()
                        p.grad = g
                    Mvs = list(torch.autograd.grad(grads, params, vs))
                    for p, g in zip(params, grads):
                        p.grad = p.prev_grad
                elif mat_type == 'fisher' or 'fisher_without_damping':
                    vs = ParamVector(params,vs)
                    fisher_config = FisherConfig(fisher_type=FISHER_MC,fisher_shapes=SHAPE_LAYER_WISE,loss_type=LOSS_CROSS_ENTROPY,data_size=self.config.data_size,fvp_attr='fvp',ignore_modules=self.config.ignore_modules)
                    fisher_maker = get_fisher_maker(model=self.model,config=fisher_config)
                    fisher_maker.setup_model_call(self._model_fn,*self._model_args)
                    fisher_maker.setup_loss_call(self._loss_fn, *self._loss_fn_args)
                    fvp_fn = fisher_maker._get_fvp_fn()
                    Mvs=fvp_fn(vec=vs)
                    vs,Mvs=list(vs.values()),list(Mvs.values())
                    Mvs = damp_vector(Mvs,vs,damping)

            if mat_type == 'numerical_check':
                import copy
                Mvs = copy.deepcopy(vs)
                self.precondition(vectors = ParamVector(params,Mvs),use_inv=False)

            vs,Mvs = list(vs),list(Mvs)
            self._criterion(vs, Mvs, criterion_type=criterion_types)

        if mat_type != 'fisher_without_damping':
            self.remove_damping()

        for module in self.module_dict.children():
            for ctype in criterion_types:
                ctype = str(ctype)
                criterion_dic['criterion'+ctype][self.module_inverse_dict[module]] = getattr(module,'criterion'+ctype) / n_samples
                criterion_dic['SGD_criterion'+ctype][self.module_inverse_dict[module]] = getattr(module,'criterion_sgd'+ctype) / n_samples
                criterion_dic['SGD_ratio_criterion'+ctype][self.module_inverse_dict[module]] = getattr(module,'criterion'+ctype) /  (getattr(module,'criterion_sgd'+ctype) +1e-24)

        for ctype in criterion_types:
            ctype = str(ctype)
            if ctype!=11 and ctype!=21:
                criterion_dic['criterion'+ctype]['all'] = getattr(self,'criterion'+ctype)
                criterion_dic['SGD_criterion'+ctype]['all'] = getattr(self,'criterion_sgd'+ctype)
                criterion_dic['SGD_ratio_criterion'+ctype]['all'] = getattr(self,'criterion'+ctype) / (getattr(self,'criterion_sgd'+ctype)+1e-24)

        return criterion_dic

    @torch.no_grad()
    def _criterion(self, dxs: List[Tensor], dgs: List[Tensor],criterion_type = [1,2,3,4]):
        ctensor_dic = {}
        for ctype in criterion_type:
            ctensor_dic[ctype] = torch.tensor([0]).to('cuda')
            ctensor_dic[-ctype] = torch.tensor([0]).to('cuda')

        for module in self.module_dict.children():
            Gx = dxs.pop(0)
            Gg = dgs.pop(0)
            Gxbias,Ggbias = None,None
            if isinstance(module, nn.Conv2d):
                Gx = Gx.flatten(start_dim=1)
                Gg = Gg.flatten(start_dim=1)

            if module.bias is not None:
                Gxbias = dxs.pop(0)
                Ggbias = dgs.pop(0)
            preGx = self.vector_precond(module,vec=Gx,vec_bias=Gxbias,inv=True)
            preGg = self.vector_precond(module,vec=Gg,vec_bias=Ggbias,inv=False)
            if module.bias is not None:
                Gx = torch.cat([Gx, Gxbias.unsqueeze(-1)], dim=1)
                Gg = torch.cat([Gg, Ggbias.unsqueeze(-1)], dim=1)

            cos = nn.CosineSimilarity(dim=0)
            for ctype in criterion_type:
                if ctype == 1:
                    criteria = parameters_to_vector(Gg) - parameters_to_vector(preGx)
                    criteria_sgd = parameters_to_vector(Gg) - parameters_to_vector(Gx)
                elif ctype == 2:
                    criteria = parameters_to_vector(preGg) - parameters_to_vector(Gx)
                    criteria_sgd = parameters_to_vector(Gg) - parameters_to_vector(Gx)
                elif ctype == 3:
                    #print('dθ P^-1 dθ',parameters_to_vector(Gx).dot(parameters_to_vector(preGx)))
                    #print('dg P    dg',parameters_to_vector(Gg).dot(parameters_to_vector(preGg)))
                    criteria = parameters_to_vector(Gx).dot(parameters_to_vector(preGx)) + parameters_to_vector(Gg).dot(parameters_to_vector(preGg))
                    criteria_sgd = parameters_to_vector(Gx).dot(parameters_to_vector(Gx)) + parameters_to_vector(Gg).dot(parameters_to_vector(Gg))
                elif ctype == 4:
                    criteria = parameters_to_vector(Gg/torch.norm(Gg)) - parameters_to_vector(preGx/torch.norm(preGx))
                    criteria_sgd = parameters_to_vector(Gg/torch.norm(Gg)) - parameters_to_vector(Gx/torch.norm(Gx))
                elif ctype == 5:
                    criteria = parameters_to_vector(preGg/torch.norm(preGg)) - parameters_to_vector(Gx/torch.norm(preGg))
                    criteria_sgd = parameters_to_vector(Gg/torch.norm(Gg)) - parameters_to_vector(Gx/torch.norm(Gx))

                elif ctype == 0:
                    criteria = cos(parameters_to_vector(Gx), parameters_to_vector(preGx))
                    criteria_sgd = cos(parameters_to_vector(Gg), parameters_to_vector(preGg))
                elif ctype == 11:
                    criteria = cos(parameters_to_vector(Gg), parameters_to_vector(preGx))
                    criteria_sgd = cos(parameters_to_vector(Gg), parameters_to_vector(Gx))
                elif ctype == 21:
                    criteria = cos(parameters_to_vector(preGg), parameters_to_vector(Gx))
                    criteria_sgd = cos(parameters_to_vector(Gg), parameters_to_vector(Gx))

                v1= getattr(module,'criterion'+str(ctype))
                v2 = getattr(module,'criterion_sgd'+str(ctype))
                setattr(module, 'criterion'+str(ctype), v1 + float(torch.norm(criteria)))
                setattr(module, 'criterion_sgd'+str(ctype), v2 + float(torch.norm(criteria_sgd)))

                ctensor_dic[ctype]  = torch.cat([ctensor_dic[ctype],  torch.flatten(criteria)])
                ctensor_dic[-ctype] = torch.cat([ctensor_dic[-ctype], torch.flatten(criteria_sgd)])

        for ctype in criterion_type:
            setattr(self,'criterion'+str(ctype),torch.norm(ctensor_dic[ctype]) )
            setattr(self,'criterion_sgd'+str(ctype),torch.norm(ctensor_dic[-ctype]))

        assert len(dxs) == 0
        assert len(dgs) == 0



def parameters_to_vector(parameters: Iterable[Tensor]) -> Tensor:
    # torch.nn.utils.parameters_to_vector uses param.view(-1) which doesn't work
    # with non-contiguous parameters
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def inverse_dict(d):
    return {v:k for k,v in d.items()}

def damp_vector(Fvs,vs,damping):
    vec = []
    Fvs = list(Fvs)
    vs = list(vs)
    for i in range(len(Fvs)):
        vec.append(Fvs[i] + damping * vs[i])
    return vec
