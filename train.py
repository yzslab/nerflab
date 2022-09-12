from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from internal.models import NeRF as NeRFModel
from internal.datasets.nerfpl import llff as nerfpl_llff, blender as nerfpl_blender
import internal.options

parameters = internal.options.get_parameters()

# train_dataset, test_dataset, val_dataset = internal.datasets.llff.get_llff_dataset(
#     "/mnt/x/NeRF-Data/nerf_dataset/nerf_llff_data/fern", 8, 8)
# train_dataset, test_dataset, val_dataset = internal.datasets.blender.get_blender_dataset(
#     "/mnt/x/NeRF-Data/nerf_dataset/nerf_synthetic/lego", True, True)

if parameters.dataset_type == "llff":
    train_dataset = nerfpl_llff.LLFFDataset(parameters.dataset_path, spheric_poses=True)
    noise_std = 1e0  # important for training real world dataset
    decay_step = [10, 20]
elif parameters.dataset_type == "blender":
    train_dataset = nerfpl_blender.BlenderDataset(parameters.dataset_path, img_wh=(400, 400))
    noise_std = 0
    decay_step = [2, 4, 8]
else:
    raise ValueError(f"unsupported dataset type: {parameters.dataset_type}")


hparams = {
    "batch": 1024,
    "chunk": 32768,
    "n_coarse_samples": 64,
    "n_fine_samples": 128,

    "white_background": False,

    "density_layers": 8,
    "density_layer_units": 256,
    "color_layers": 1,
    "color_layer_units": 128,
    "skips": [4],

    "encoding": "pe",
    "pe_location_n_freq": 10,
    "pe_direction_n_freq": 4,

    "perturb": 1.0,
    "noise_std": noise_std,

    "lrate": 5e-4,
    "optimizer": "adam",
    "lr_scheduler": "steplr",
    "momentum": 0.9,
    "weight_decay": 0,
    "decay_step": decay_step,
    "decay_gamma": 0.5,
    "warmup_epochs": 0,
}

train_loader = DataLoader(train_dataset, batch_size=hparams["batch"], shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
# val_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

callbacks = []

logger = TensorBoardLogger(save_dir=parameters.log_dir, name=parameters.exp_name, default_hp_metric=False)

nerf_model = NeRFModel(hparams)
trainer = pl.Trainer(
    max_epochs=1,
    callbacks=callbacks,
    logger=logger,
    accelerator='gpu',
    devices=1,
    benchmark=True,
    profiler="simple",
    enable_model_summary=True,
)
trainer.fit(model=nerf_model, train_dataloaders=train_loader)
