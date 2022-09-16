from torch.utils.data import DataLoader
import pytorch_lightning as pl
from internal.models import NeRF as NeRFModel
import internal.options

parameters = internal.options.get_parameters()
train_dataset, test_dataset, val_dataset, extra_hparams = internal.options.get_dataset_by_parameters(parameters)

model = NeRFModel.load_from_checkpoint(parameters.load_ckpt, log_dir=parameters.log_dir, exp_name=parameters.exp_name)
trainer = pl.Trainer(
    accelerator=parameters.accelerator,
    devices=parameters.n_device,
)
predictions = trainer.predict(model, DataLoader(test_dataset, batch_size=1, shuffle=False),
                              ckpt_path=parameters.load_ckpt)

for i, val in enumerate(predictions):
    print(f"#{i} loss: {val['val/loss']}, psnr: {val['val/psnr'][0]}")
