import internal.common
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from internal.lightning_modules.nerf import NeRF as NeRFModel
from pytorch_lightning.strategies.ddp import DDPStrategy
import internal.arguments

# parser arguments and config files
arguments, hparams = internal.arguments.get_arguments()

# load dataset
train_dataset, test_dataset, val_dataset = internal.arguments.get_dataset_by_hparams(hparams)

# create dataloader
train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=hparams["dataloader_num_workers"])
# test_loader = DataLoader(test_dataset, batch_size=hparams["batch"], shuffle=False)
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=hparams["dataloader_num_workers"])

callbacks = [
    # LearningRateMonitor(logging_interval="step"),
    ModelCheckpoint(
        dirpath=f"ckpts/{arguments.exp_name}",
        monitor="step",
        mode="max",
        every_n_train_steps=1000,
        save_top_k=10,
        filename='{step:06d}',
    ),
    ModelCheckpoint(
        dirpath=f"ckpts/{arguments.exp_name}",
        monitor="epoch",
        mode="max",
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
    precision=16 if hparams["network_type"] == "tcnn_ff" else 32,
)
trainer.fit(model=nerf_model, train_dataloaders=train_loader, val_dataloaders=val_loader, **trainer_extra_parameters)
