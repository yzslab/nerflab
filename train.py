from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from internal.models import NeRF as NeRFModel
import internal.datasets.blender
# from internal.datasets.nerfpl import llff as nerfpl_llff, blender as nerfpl_blender
import internal.options

parameters = internal.options.get_parameters()
train_dataset, test_dataset, val_dataset, extra_hparams = internal.options.get_dataset_by_parameters(parameters)

hparams = {
    "batch": 1024,
    "chunk": 32768,
    "n_coarse_samples": 64,
    "n_fine_samples": 128,

    # "white_background": white_background,

    "density_layers": 8,
    "density_layer_units": 256,
    "color_layers": 1,
    "color_layer_units": 128,
    "skips": [4],

    "encoding": "pe",
    "pe_location_n_freq": 10,
    "pe_direction_n_freq": 4,

    "perturb": 1.0,
    # "noise_std": noise_std,

    "lrate": 5e-4,
    "optimizer": "adam",
    "lr_scheduler": "steplr",
    "momentum": 0.9,
    "weight_decay": 0,
    # "decay_step": decay_step,
    "decay_gamma": 0.5,
    "warmup_epochs": 0,
}

# override hparams from extra_hparams
for k in extra_hparams:
    hparams[k] = extra_hparams[k]

train_loader = DataLoader(train_dataset, batch_size=hparams["batch"], shuffle=True)
#test_loader = DataLoader(test_dataset, batch_size=hparams["batch"], shuffle=False)
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

callbacks = [
    # LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(dirpath=f"{parameters.log_dir}/{parameters.exp_name}/ckpts", filename='{epoch:02d}')
]

logger = TensorBoardLogger(save_dir=parameters.log_dir, name=parameters.exp_name, default_hp_metric=False)

nerf_model = NeRFModel(hparams)
trainer = pl.Trainer(
    max_epochs=parameters.n_epoch,
    callbacks=callbacks,
    logger=logger,
    accelerator=parameters.accelerator,
    devices=parameters.n_device,
    num_sanity_val_steps=1
)
trainer.fit(model=nerf_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
