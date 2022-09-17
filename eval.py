from torch.utils.data import DataLoader
import pytorch_lightning as pl
from internal.models import NeRF as NeRFModel
import internal.arguments

arguments = internal.arguments.get_arguments()
train_dataset, test_dataset, val_dataset, extra_hparams = internal.arguments.get_dataset_by_arguments(arguments)

model = NeRFModel.load_from_checkpoint(
    arguments.load_ckpt,
    log_dir=arguments.log_dir,
    exp_name=arguments.exp_name,
    perturb=0.,
    noise_std=0.
)
trainer = pl.Trainer(
    accelerator=arguments.accelerator,
    devices=arguments.n_device,
    logger=internal.arguments.get_logger_by_arguments(arguments),
)
trainer.predict(model, DataLoader(test_dataset, batch_size=1, shuffle=False),
                ckpt_path=arguments.load_ckpt)
