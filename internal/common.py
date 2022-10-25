import os
import pytorch_lightning as pl

# setup seed
seed = 0
pl.seed_everything(seed, workers=True)
os.environ['PYTHONHASHSEED'] = str(seed)
print(f"everything seeded with {seed}")
