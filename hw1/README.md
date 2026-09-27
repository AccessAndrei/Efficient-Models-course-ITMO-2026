# HW-1

The four functions `FLOPs(S, B)`, `Memory(S, B)`, `Latency(S, B, θ)` and `Energy(S, B, θ)` are derived by hand in `hw1_handwritten.pdf`, implemented in `equations.py`, and checked against measurements of the real network on a Tesla T4.

## Environment

| | |
|---|---|
| GPU | Tesla T4 (Google Colab) |
| NVIDIA driver | 580.82.07 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| cuDNN | 9.19.0 |
| Python | 3.13 |

Package versions match the Colab runtime (actual one) and are pinned in `requirements.txt`.

## Model

`models.py` holds the network from the assignment: six convolutions (7×7 s2 3→32, 5×5 32→64, 3×3 s2 64→128, 1×1 128→256, 3×3 s2 256→256, 1×1 256→512), a 3×3 s2 max-pool after the first one, then global average pooling, Linear 512→256, ReLU and Linear 256→100.

**Assumption:** every convolution is followed by BatchNorm and then ReLU (Conv → BN → ReLU). Convolutions have `bias=False`, and ReLU is `inplace=True`. Nothing is trained, so BatchNorm keeps its default running statistics (mean 0, variance 1). In `eval()` mode it computes `x · a + b` per element, but it still runs its own kernel and writes a new tensor.

## How to reproduce

1. **Measurements (GPU).** In Colab, choose Runtime → Change runtime type → T4 GPU. Upload `models.py` and `measure.py`, then run:
   ```
   !python measure.py
   ```
   This writes `results/measurements.csv`.
2. **Calibration and figures (CPU is enough):**
   ```
   python -m venv .venv
   .venv/bin/pip install numpy==2.1.3 pandas==2.2.3 scipy==1.16.3 matplotlib==3.10.0
   .venv/bin/python calibrate.py   # writes results/theta.json
   .venv/bin/python plots.py       # writes results/figures/*.png
   ```

| File | Purpose |
|---|---|
| `models.py` | the network |
| `equations.py` | `flops()`, `memory()`, `latency()`, `energy()`, plus `bytes_moved()`; all accept NumPy arrays |
| `measure.py` | measures latency, peak memory and energy for all 132 configurations |
| `calibrate.py` | fits θ on the 63 base-grid points and reports the error on the 69 validation points |
| `plots.py` | draws the figures |

## Measurement protocol

- **Flags:** `cudnn.benchmark = False`, TF32 disabled, `model.cuda().eval()`, `torch.inference_mode()`. Inputs are random tensors.
- **Grid:**
  - S ∈ {32, 64, 128, 224, 256, 384, 512} plus 4 random multiples of 16: **80, 208, 304, 320**;
  - B ∈ {1, 2, 4, …, 256} plus 3 random non-powers of two: **111, 133, 139** (seed 0);
  - 11 × 12 = 132 configurations. A point is in the validation set if its S or its B is one of the random values: 69 validation points, 63 base points.
- **Latency:** 5 warm-up passes, then the median of 20 passes. Each pass is timed with `time.perf_counter()` between two `torch.cuda.synchronize()` calls.
- **Memory:** `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` and one pass. Weights and the input are already allocated, so they are included.
- **Energy:** the NVML total-energy counter of the whole GPU. The network runs back to back for 1 s, and the energy difference is divided by the number of passes.

## Equations

S is the image size and B the batch size; all numbers are FP32, 4 bytes each.

| Function | Result |
|---|---|
| FLOPs (1 MAC = 2 FLOPs; conv, BN, ReLU, pooling and linear layers) | 17 793·B·S² + 313 600·B |
| Bytes moved (every tensor read and written once) | 532·B·S² + 8 592·B + 4 181 264 |
| Memory (weights + all activations at once; see below) | 4 181 312 + 4·B·(47·S² + 868) bytes |
| Latency | Σᵢ [t_launch + max(FLOPsᵢ / P, Bytesᵢ / BW)] over the 23 kernels |
| Energy | e_flop · FLOPs + e_byte · Bytes + P_static · Latency |

- **Memory convention.** It was confirmed at the chat that the memory model should sum all activations, as if they were alive at the same time. The in-place ReLU and `flatten` allocate nothing.
- **Latency parameters:** θ = (t_launch, P, BW).
- **Energy parameters:** θ_energy = (e_flop, e_byte, P_static); `energy()` also receives θ, because it uses `latency()`.

At S = 224, B = 1 the formulas give 893 MFLOPs, 30.9 MB moved and 13.0 MiB of memory.

## Results

**Fitted parameters** (`results/theta.json`):

| Parameter | Fitted | T4 datasheet |
|---|---|---|
| t_launch | 27.0 µs | — |
| P | 6.88 TFLOPS | 8.1 TFLOPS |
| BW | 92.6 GB/s | 320 GB/s |
| e_flop | 7.12 pJ | — |
| e_byte | ≈ 0 pJ | — |
| P_static | 50.7 W | 70 W (TDP) |

**Prediction error**, |predicted / measured − 1|:

| | Latency: mean / max | Energy: mean / max | Memory: mean / max |
|---|---|---|---|
| Base grid (63 points) | 20.7% / 83.4% | 17.3% / 48.8% | 69.8% / 130.1% |
| Validation (69 points) | 25.2% / 61.4% | 22.4% / 47.3% | 76.8% / 114.1% |

**OOM:** none. No configuration ran out of memory, and the memory formula predicts none either: its maximum is 11.75 GiB at S = 512, B = 256, below the T4's 15 GiB.

**Figures** (`results/figures/`):

| File | Content |
|---|---|
| `latency.png`, `energy.png`, `memory.png` | the quantity against B, one colour per S; lines are predictions, dots are base points, crosses are validation points |
| `parity.png` | predicted against measured for all 132 points |
| `*_error.png` | signed error for every (S, B) cell |

## Discussion

- **Launch-bound.** For small S and B, latency stays flat at about 0.7–0.8 ms whatever the input. The fitted t_launch = 27 µs times 23 kernels gives this plateau. Most of it is Python and PyTorch dispatch overhead per operator probably.
- **Compute-bound and memory-bound.** For large configurations, latency scales as B·S², as the equations predict:
  - doubling B at S = 512 multiplies the time by 1.97;
  - doubling S at B = 256 multiplies it by 3.92.
- **Two regimes inside the network.** By arithmetic intensity (FLOPs per byte), every convolution is compute-bound (43–267 FLOPs/byte against a T4 ridge point of about 25). BatchNorm, ReLU and pooling are memory-bound (0.125–0.4 FLOPs/byte). The memory-bound layers are only 0.5% of the FLOPs (memory limited layers), yet the model gives them roughly a third of the time.

**Where the latency model breaks.**
1. **A step between B = 64 and B = 111.** Beyond B = 64 the time per image jumps: from 0.34 to 0.50 ms at S = 224 and from 1.73 to 2.51 ms at S = 512 (×1.45–1.7). The equations are smooth in B, so after calibration they underestimate every B ≥ 111 point by 15–45% and overestimate the middle of the grid.
   - Peak memory at S = 32 also jumps at the same boundary, from 18.7 MB to 53 MB.
   - My hypothesis is that with `cudnn.benchmark = False`, cuDNN's heuristics switch the convolutions to a different algorithm, with a different workspace, for batches above 64. 
2. **Kernel launch overlaps with GPU work.** Between the launch-bound plateau and the linear regime (B = 1–4, S ≥ 208), the model overestimates by up to 83%. It adds t_launch to every kernel's GPU time. In reality the CPU enqueues the next kernel while the GPU still runs the previous one, so each kernel costs closer to max(t_launch, GPU time) than to their sum.
3. **P and BW are not separately identifiable.** In the large regime, latency ≈ 23·t_launch + (17 712 / P + 384 / BW)·B·S². The measurements only fix the sum in brackets, so the fit can trade P against BW. The fitted BW of 92.6 GB/s (29% of the datasheet) most likely absorbs part of the convolutions' inefficiency. It should not be read as the T4's real bandwidth.

**Energy.**
- Measured power ranges from about 33 W in the launch-bound corner to 60–65 W for the largest configurations. So the GPU burns substantial power even when it has almost no work, which the P_static · Latency term captures.
- A single constant P_static (50.7 W) cannot describe both ends of that range, and the energy error inherits the latency error pattern, including the B = 64 step.
- The fit set e_byte to about 0. Bytes already enter through Latency (the Bytes / BW term), so the two terms are almost collinear and the byte term adds nothing.
- Energy per image is lowest around B = 16–64 and rises again beyond B = 64, following the latency step. For example, at S = 224 it is 51 mJ at B = 1, 23 mJ at B = 64 and 33 mJ at B = 128. Below the step, the static power and the weight reads are shared by more images; above it, each image simply takes longer.

**Memory.** The convention (all activations alive at once) is an upper bound.
- For large inputs it overestimates by about 2×. PyTorch frees each activation as soon as it is no longer referenced, so the real peak is at the first BatchNorm: the input, the conv1 output and the BN1 output are alive together, 19·B·S² numbers, or 76 bytes per B·S². The measured 88–101 bytes per B·S² is close to that.
- For small inputs it underestimates by up to 77%: about 15–20 MB is measured against 4 MB of weights, most likely cuDNN and cuBLAS workspaces, which the formula ignores.
