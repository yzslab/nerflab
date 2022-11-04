# NeRF Laboratory
## Setup
- Environment setup
```bash
conda create -n nerflab python=3.8 pip
conda activate nerflab

git clone https://github.com/yzslab/nerflab.git
cd nerflab

pip install -r requirements.txt
```
- Optionally: install tiny-cuda-nn PyTorch extension (Only require if you need tiny-cuda-nn accelerate)
```bash
export CUDA_VERSION="11.3"
export MAKEFLAGS="-j$(nproc)"
export PATH="/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:/usr/lib/wsl/lib/:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${LD_LIBRARY_PATH}:${LIBRARY_PATH}"

git submodule sync --recursive
git submodule update --init --recursive

pushd dependencies/tiny-cuda-nn/bindings/torch
python setup.py install
popd
```
- Dataset preparation
  - [NeRF llff & synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
  - [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip)
## Training
### Examples
- Blender
```bash
python train.py \
  --config configs/blender.yaml \
  --dataset-path nerf_dataset/nerf_synthetic/lego \
  --exp-name lego
```
- LLFF
```bash
python train.py \
  --config configs/llff.yaml \
  --dataset-path nerf_dataset/nerf_llff_data/fern \
  --exp-name fern
```
- Accelerate with tiny-cuda-nn cutlass MLP
```bash
python train.py \
  --config \
    configs/blender.yaml \
    configs/tcnn_cutlass.yaml \
  --dataset-path nerf_dataset/nerf_synthetic/lego \
  --exp-name lego-tcnn-cutlass
```
- Multiresolution Hash Encoding implemented by tiny-cuda-nn
```bash
python train.py \
  --config \
    configs/blender.yaml \
    configs/tcnn_hash.yaml \
  --dataset-path nerf_dataset/nerf_synthetic/lego \
  --exp-name lego-tcnn-hash
```
### Set hyperparameters via command line
- Use `--config-values`:
```bash
python train.py \
  --config configs/llff.yaml \
  --config-values \
    'llff_down_sample_factor=8' \
    'batch_size=4096' \
    'chunk_size=65536' \
  --dataset-path nerf_dataset/nerf_llff_data/fern \
  --exp-name fern
```

## Evaluation (Rendering)
- Example
```bash
python eval.py \
  --load-ckpt ./ckpts/lego/YOUR_CKPT_FILENAME.ckpt
```
- Or more specifically
```bash
python eval.py \
  --config configs/blender.yaml \
  --dataset-path nerf_dataset/nerf_synthetic/lego \
  --load-ckpt ./ckpts/lego/YOUR_CKPT_FILENAME.ckpt
```
## Related Documents
- [bmild/nerf 源码注释](https://www.yuque.com/docs/share/01c0c96c-fdc1-472e-acf4-a83aa59f5c6f)
- [实现细节](https://www.yuque.com/docs/share/6d2e30ca-963f-439c-b5b5-776954e6507f)
## To-Do List
- [x] Implement original positional encoding NeRF using PyTorch Lightning (1 week)
- [ ] Implement integrated positional encoding from Mip-NeRF (3 days)
- [ ] Add unstructured image collections support (NeRF in the Wild & Hallucinated Neural Radiance Fields in the Wild) (2 weeks)
- [ ] Implement multi-block merge like Block-NeRF (2 weeks)
- [ ] Add camera self-calibration algorithms from SCNeRF (1 week)
## Known issues
- Sometimes the coarse or fine network can't be optimized when training on blender dataset.
  - Related issue
    - https://github.com/bmild/nerf/issues/29
    - https://github.com/kwea123/nerf_pl/issues/75
    - https://github.com/kwea123/nerf_pl/issues/51
  - Setting noise_std to 1.0 seems can solve the problem. 
## References
- [bmild/nerf](https://github.com/bmild/nerf)
- [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)
- [nerf_pl](https://github.com/kwea123/nerf_pl)
- [【AI講壇】程式碼導讀 - NeRF](https://youtu.be/SoEehTR2MiM), [【AI講壇】程式碼導讀 - NeRF (2)](https://youtu.be/kh_hxFnuQNI)
- [Ray-Tracing](https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/definition-ray)