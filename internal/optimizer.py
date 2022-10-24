import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LambdaLR


def get_optimizer(models: list, hparams: dict):
    parameters = []
    for model in models:
        parameters += list(model.parameters())

    extra_optimizer_params = {
        "eps": 1e-8,
    }

    if "betas" in hparams:
        extra_optimizer_params["betas"] = (hparams["betas"][0], hparams["betas"][1])
    if "eps" in hparams:
        extra_optimizer_params["eps"] = float(hparams["eps"])

    optimizer = torch.optim.Adam(parameters, lr=hparams["lrate"], **extra_optimizer_params)

    return optimizer


def get_scheduler(optimizer, hparams: dict):
    eps = 1e-8
    if hparams["lr_scheduler"] == 'steplr':
        scheduler = MultiStepLR(optimizer, milestones=hparams["decay_step"],
                                gamma=hparams["decay_gamma"])
    elif hparams["lr_scheduler"] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams["n_epoch"], eta_min=eps)
    elif hparams["lr_scheduler"] == 'exponential':
        if hparams["scheduler_interval"] == "step":
            decay_rate = 0.1
            lrate_decay = hparams["lrate_decay"]
            decay_steps = lrate_decay * 1000

            def scheduler_func(step):
                return decay_rate ** (step / decay_steps)

            scheduler = LambdaLR(optimizer, scheduler_func)
            scheduler = {
                "scheduler": scheduler,
                "interval": "step",
            }
        elif hparams["scheduler_interval"] == "epoch":
            scheduler = LambdaLR(optimizer,
                                 lambda epoch: (1 - epoch / hparams["n_epoch"]) ** hparams["scheduler_exponent"])
        else:
            raise ValueError("unsupported scheduler interval type")
    else:
        raise ValueError('scheduler not recognized')

    return scheduler
