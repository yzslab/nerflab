from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from internal.models import NeRF as NeRFModel
import internal.arguments

arguments = internal.arguments.get_arguments()
train_dataset, test_dataset, val_dataset, extra_hparams = internal.arguments.get_dataset_by_arguments(arguments)

hparams = {
    "batch": arguments.batch_size,
    "chunk": arguments.chunk_size,
    "n_coarse_samples": 64,
    "n_fine_samples": 128,

    "white_background": arguments.white_bkgd,

    "density_layers": 8,
    "density_layer_units": 256,
    "color_layers": 1,
    "color_layer_units": 128,
    "skips": [4],

    "encoding": "pe",
    "pe_location_n_freq": 10,
    "pe_direction_n_freq": 4,

    "perturb": arguments.perturb,
    "noise_std": arguments.noise_std,

    "lrate": 5e-4,
    "optimizer": "adam",
    "lrate_decay": arguments.lrate_decay,
    "lr_scheduler": "exponential",
    # "momentum": 0.9,
    # "weight_decay": 0,
    # # "decay_step": decay_step,
    # "decay_gamma": 0.5,
    # "warmup_epochs": 0,
}

# override hparams from extra_hparams
for k in extra_hparams:
    hparams[k] = extra_hparams[k]

train_loader = DataLoader(train_dataset, batch_size=hparams["batch"], shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=hparams["batch"], shuffle=False)
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

callbacks = [
    # LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(
        dirpath=f"ckpts/{arguments.exp_name}",
        every_n_train_steps=500,
        save_top_k=-1,
        filename='{step:05d}',
    ),
    ModelCheckpoint(
        dirpath=f"ckpts/{arguments.exp_name}",
        save_top_k=-1,
        filename='{epoch:02d}-{step:d}',
    )
]

logger = internal.arguments.get_logger_by_arguments(arguments)

trainer_extra_parameters = {}
if arguments.load_ckpt is None:
    nerf_model = NeRFModel(hparams)
else:
    nerf_model = NeRFModel.load_from_checkpoint(arguments.load_ckpt)
    trainer_extra_parameters["ckpt_path"] = arguments.load_ckpt

trainer = pl.Trainer(
    max_epochs=arguments.n_epoch,
    callbacks=callbacks,
    logger=logger,
    accelerator=arguments.accelerator,
    devices=arguments.n_device,
    num_sanity_val_steps=1,
    limit_val_batches=3,
)
trainer.fit(model=nerf_model, train_dataloaders=train_loader, val_dataloaders=val_loader, **trainer_extra_parameters)
