import os.path

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from internal.lightning_modules.nerf import NeRF as NeRFModel
import internal.arguments

arguments, hparams = internal.arguments.get_arguments()

## set evaluation argument
eval_arguments = {
    "eval_name": arguments.eval_name
}
# set eval name
if eval_arguments["eval_name"] is None:
    eval_arguments["eval_name"] = os.path.basename(arguments.load_ckpt)
# set experiment name
if arguments.exp_name is not None:
    eval_arguments["exp_name"] = arguments.exp_name

model = NeRFModel.load_from_checkpoint(
    arguments.load_ckpt,
    log_dir=arguments.log_dir,
    perturb=0.,
    noise_std=0.,
    **eval_arguments,
)
trainer = pl.Trainer(
    accelerator=arguments.accelerator,
    devices=arguments.n_device,
    strategy=DDPStrategy(find_unused_parameters=False) if arguments.n_device > 1 else None,
    logger=False,
)

hparams = model.hparams
train_dataset, test_dataset, val_dataset = internal.arguments.get_dataset_by_hparams(hparams)

trainer.predict(model, DataLoader(test_dataset, batch_size=1, shuffle=False),
                ckpt_path=arguments.load_ckpt)
