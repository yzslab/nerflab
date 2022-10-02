from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from internal.lightning_modules.nerf import NeRF as NeRFModel
from pytorch_lightning.strategies.ddp import DDPStrategy
import internal.arguments

arguments, hparams = internal.arguments.get_arguments()

train_dataset, test_dataset, val_dataset = internal.arguments.get_dataset_by_hparams(hparams)

# hparams = {
#     "batch": arguments.batch_size,
#     "chunk": arguments.chunk_size,
#     "n_coarse_samples": 64,
#     "n_fine_samples": 128,
#
#     "white_background": arguments.white_bkgd,
#
#     "density_layers": 8,
#     "density_layer_units": 256,
#     "color_layers": 1,
#     "color_layer_units": 128,
#     "skips": [4],
#
#     "encoding": "pe",
#     "pe_location_n_freq": 10,
#     "pe_direction_n_freq": 4,
#
#     "perturb": arguments.perturb,
#     "noise_std": arguments.noise_std,
#
#     "lrate": 5e-4,
#     "optimizer": "adam",
#     "lrate_decay": arguments.lrate_decay,
#     "lr_scheduler": "exponential",
#     # "momentum": 0.9,
#     # "weight_decay": 0,
#     # # "decay_step": decay_step,
#     # "decay_gamma": 0.5,
#     # "warmup_epochs": 0,
# }

train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=hparams["batch"], shuffle=False)
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

callbacks = [
    # LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(
        dirpath=f"ckpts/{arguments.exp_name}",
        monitor="step",
        every_n_train_steps=1000,
        save_top_k=10,
        filename='{step:06d}',
    ),
    ModelCheckpoint(
        dirpath=f"ckpts/{arguments.exp_name}",
        monitor="epoch",
        save_top_k=10,
        filename='{epoch:02d}-{step:d}',
    )
]

logger = internal.arguments.get_logger_by_arguments(arguments)

trainer_extra_parameters = {}
if arguments.load_ckpt is None:
    nerf_model = NeRFModel(hparams)
else:
    nerf_model = NeRFModel.load_from_checkpoint(arguments.load_ckpt, **hparams)
    trainer_extra_parameters["ckpt_path"] = arguments.load_ckpt

trainer = pl.Trainer(
    max_epochs=arguments.n_epoch,
    callbacks=callbacks,
    logger=logger,
    accelerator=arguments.accelerator,
    devices=arguments.n_device,
    strategy=DDPStrategy(find_unused_parameters=False) if arguments.n_device > 1 else None,
    num_sanity_val_steps=1,
    limit_val_batches=3,
)
trainer.fit(model=nerf_model, train_dataloaders=train_loader, val_dataloaders=val_loader, **trainer_extra_parameters)
