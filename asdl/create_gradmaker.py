import torch
from torch import nn
import torch_optimizer as optim
from .precondition import *
from .optimizers import *

OPTIM_SGD = 'sgd'
OPTIM_ADAMW = 'adamw'
OPTIM_ADAM = 'adam'
OPTIM_ADAM_2 = 'adam_v2'
OPTIM_SHAMPOO='shampoo'
OPTIM_KFAC_MC = 'kfac_mc'
OPTIM_KFAC_EMP = 'kfac_emp'
OPTIM_NOISY_KFAC_MC = 'noisy_kfac_mc'
OPTIM_SMW_NGD = 'smw_ngd'
OPTIM_FULL_PSGD = 'full_psgd'
OPTIM_KRON_PSGD = 'psgd'
OPTIM_NEWTON = 'newton'
OPTIM_ABS_NEWTON = 'abs_newton'
OPTIM_KBFGS = 'kbfgs'
OPTIM_CURVE_BALL = 'curve_ball'
OPTIM_SENG = 'seng'
OPTIM_ADAHESSIAN = 'adahessian'
OPTIM_SWATS = 'swats'
OPTIM_FOOF = 'foof'
OPTIM_BOOB = 'boob'
OPTIM_NGD_LAYER_WISE = 'ngd_layerwise'
OPTIM_NGD_FULL = 'ngd_full'
OPTIM_LARS = 'lars'
OPTIM_LAMB = 'lamb'
OPTIM_KFAC_EMP_ADAM = 'kfac_emp_adam'
OPTIM_KFAC_MC_ADAM = 'kfac_mc_adam'
OPTIM_PSGD_ADAM = 'psgd_adam'
OPTIM_SHAMPOO_ADAM = 'shampoo_adam'

def create_grad_maker(model,args):
    args.ignore_module_name.extend([nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm])

    if args.optim == OPTIM_ADAMW or args.base_optim == OPTIM_ADAMW or args.optim == OPTIM_KFAC_MC_ADAM or args.optim == OPTIM_KFAC_EMP_ADAM or args.optim == OPTIM_SHAMPOO_ADAM or args.optim == OPTIM_PSGD_ADAM:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps = args.opt_eps)
    elif args.optim == OPTIM_ADAM or args.base_optim == OPTIM_ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps = args.opt_eps)
    elif args.optim == OPTIM_ADAHESSIAN:
        optimizer = optim.Adahessian(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps = args.damping, hessian_power=1.0)
    elif args.optim == OPTIM_SWATS:
        optimizer = optim.SWATS(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps = args.damping)
    elif args.optim == OPTIM_LARS or args.base_optim == OPTIM_LARS:
        optimizer = LARS(model.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == OPTIM_LAMB or args.base_optim == OPTIM_LAMB:
        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps = args.opt_eps)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)

    config = PreconditioningConfig(data_size=args.batch_size,
                                    damping=args.damping,
                                    ema_decay = args.ema_decay,
                                    preconditioner_upd_interval=args.curvature_update_interval,
                                    curvature_upd_interval=args.curvature_update_interval,
                                    precondition_modules=args.precond_module_name,
                                    ignore_modules=(args.ignore_module_name),
                                    kl_clip=args.kl_clip
                                    )

    if args.optim == OPTIM_KFAC_MC or args.optim == OPTIM_KFAC_MC_ADAM:
        grad_maker = KfacGradientMaker(model, config, zero_initialization = args.zero_initialization)
    elif args.optim == 'adam_v2':
        grad_maker = AdamGradientMaker(model, config, optimizer=optimizer)
    elif args.optim == OPTIM_KFAC_EMP or args.optim == OPTIM_KFAC_EMP_ADAM:
        grad_maker = KfacEmpGradientMaker(model, config, zero_initialization = args.zero_initialization)
    elif args.optim == OPTIM_FOOF:
        grad_maker = FOOFGradientMaker(model, config, zero_initialization = args.zero_initialization)
    elif args.optim == OPTIM_BOOB:
        grad_maker = BOOBGradientMaker(model, config, zero_initialization = args.zero_initialization)
    elif args.optim == OPTIM_NGD_FULL:
        grad_maker = FullNaturalGradientMaker(model, config, zero_initialization = args.zero_initialization)
    elif args.optim == OPTIM_NGD_LAYER_WISE:
        grad_maker = LayerWiseNaturalGradientMaker(model, config, zero_initialization = args.zero_initialization)
    elif args.optim == OPTIM_SHAMPOO or args.optim == OPTIM_SHAMPOO_ADAM:
        grad_maker = ShampooGradientMaker(model,config)
    elif args.optim == OPTIM_SMW_NGD:
        grad_maker = SmwEmpNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_FULL_PSGD:
        grad_maker = PsgdGradientMaker(model)
    elif args.optim == OPTIM_KRON_PSGD or args.optim == OPTIM_PSGD_ADAM:
        grad_maker = KronPsgdGradientMaker(model,config,precond_lr=args.precond_lr)
    elif args.optim == OPTIM_NEWTON:
        grad_maker = NewtonGradientMaker(model, config)
    elif args.optim == OPTIM_ABS_NEWTON:
        grad_maker = NewtonGradientMaker(model, config)
    elif args.optim == OPTIM_KBFGS:
        grad_maker = KronBfgsGradientMaker(model, config)
    elif args.optim == OPTIM_CURVE_BALL:
        grad_maker = CurveBallGradientMaker(model, config)
    elif args.optim == OPTIM_SENG:
        grad_maker = SengGradientMaker(model,config=config,sketching_size=args.sketching_size ,truncated_rank=args.truncated_rank)
    else:
        grad_maker = PreconditionedGradientMaker(model,config)

    return optimizer,grad_maker