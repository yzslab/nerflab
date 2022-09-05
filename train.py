import internal.datasets.llff
from torch.utils.data import DataLoader
from internal.modules import NeRF
from internal.modules import PositionalEncoding

train_set, val_set, test_set = internal.datasets.llff.get_llff_dataset(
    "/mnt/x/NeRF-Data/nerf_dataset/nerf_llff_data/fern", 4, 8)
train_set = DataLoader(train_set)
val_set = DataLoader(val_set)
test_set = DataLoader(test_set)

nerf_model = NeRF(location_encoder=PositionalEncoding(3, 10), view_direction_encoder=PositionalEncoding(3, 4))
