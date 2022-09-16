# NeRF Laboratory
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