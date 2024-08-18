# ComfyUI Flash Attention Triton

A ComfyUI node that allows you to select [Flash Attention Triton implementation](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) as sampling attention.

This implementation is approximately 20% slower than sub-quadratic attention on tested hardware, but uses less VRAM.

## Performance Comparison

Testing on a RX 6700 XT, generating a 1024x1024 image with FLUX.1-dev q4_K-S:

| Attention Method      | VRAM Usage | Speed (s/it) |
|-----------------------|------------|--------------|
| Flash Attention Triton| 8.2 GB     | 28.60        |
| Sub-Quadratic         | 9.4 GB     | 23.23        |

## Notes

- Currently, only implemented for sampling, not for VAE.
- Only compatible with FLUX models at the moment.

