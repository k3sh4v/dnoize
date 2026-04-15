# torch_directml Constraints Reference
# Hardware: AMD RX 560X 4GB | torch_directml | Python 3.12.9
# Project-agnostic. Use as base constraint reference for any PyTorch project
# targeting torch_directml (privateuseone:0 device).
# Confirmed through exhaustive testing including training and inference workloads.

---

## 1. DEVICE SETUP

```python
import torch_directml
dml  = torch_directml.device()          # privateuseone:0
name = torch_directml.device_name(0)    # e.g. "AMD Radeon RX 560X"

# Check device type:
tensor.device.type == 'privateuseone'   # True if on DML
```

---

## 2. CONFIRMED BROKEN — Never use in training (backward pass)

### Autograd / Backward Pass
- `torch.utils.checkpoint` — gradient checkpointing broken entirely, no workaround
- `F.pad` backward (FP16 or FP32) — silent wrong gradients
- FP16 sigmoid gate backward — NaN gradients
- `torch.amp.autocast("privateuseone")` — broken entirely
- GELU backward — replace with ReLU everywhere
- `aten::_foreach_lerp_` — triggered by standard AdamW, silent CPU fallback
- `repeat_interleave` backward — broken
- `ConvTranspose1d(groups=channels)` — broken backward
- `Conv1d` grad reduction at very large sequence lengths — wrong gradients
- `torch.lerp` / `aten::lerp.Scalar_out` — broken
- `torch.stft` backward — all STFT must run on CPU during training

### Optimizers
- `torch.optim.AdamW` — triggers `aten::_foreach_lerp_` → use `Adam(decouple_wd=True)`
- `LAMB` — catastrophic: `.item()` per param per step → full pipeline flush per step
- Any optimizer using `foreach` ops — verify before using

### Mixed Precision
- FP16 training — NOT viable (multiple broken backward ops, silent wrong gradients)
- FP16 inference — OK (forward pass only)
- Always train in FP32

### Other
- `buf.detach() + copy_()` for buffer updates — breaks gradient flow silently
  → Use `torch.cat` instead for causal/streaming buffer patterns
- No sync/flush/stream API in torch_directml
  → Confirmed via full recursive inspection of torch_directml module namespace

---

## 3. CONFIRMED WORKING

### Core Ops (FP32)
- Conv1d forward + backward ✓
- InstanceNorm1d ✓
- ReLU forward + backward ✓
- Sigmoid forward (inference only) ✓
- GLU via split + ReLU + mul ✓
- Element-wise: add, mul, clamp, pow, sqrt, log ✓
- torch.cat ✓
- F.pad forward FP32 (no backward) ✓
- torch.stft forward on DML (no backward) ✓
- Linear layers ✓
- LayerNorm ✓

### Optimizers
- `Adam(decouple_wd=True, eps=1e-7)` ✓
  — eps=1e-7 (vs default 1e-8) recommended for DML numerical stability
- `Lookahead(Adam(...), k=10, alpha=0.5)` ✓
  — all Lookahead ops (mul_, add_, copy_) are DML-native
- SGD with momentum ✓ (confirmed in Microsoft's own DML examples)

### Attention (inference only)
```python
# Fused inference kernel — no backward registered
y, past_k, past_v = torch_directml.multi_head_attention(
    q, k, v, n_state, n_head,
    past_key_tensor, past_value_tensor,
    mask   # integer dtype, NOT additive float mask
)
# For autoregressive KV-cache decoding only — do not use in training
```

---

## 4. PERFORMANCE CHARACTERISTICS

### Sync Points (DML↔CPU)
- Every `.item()` on a DML tensor = full pipeline flush (~88ms on RX 560X)
- Minimize `.item()` inside training loops
- Use Python float accumulation for metrics/loss smoothing — no per-batch `.item()`

### DML Command Queue
- Queue depth grows ~+1.45ms/batch per epoch without forced sync
- Force one `.item()` sync per epoch end to reset (implement as callback)

### Memory / Bandwidth
- RX 560X: 4GB VRAM, 128-bit bus, 112 GB/s — bandwidth limited, not compute limited
- Large T + Conv1d grad reduction → OOM or wrong gradients
  → Workaround: pin final Conv1d to CPU

### .item() Cost Budget
- 1 sync/epoch (queue flush): acceptable
- 1 sync/batch (loss logging): acceptable
- N syncs/batch (per-parameter): catastrophic (see LAMB)

---

## 5. ARCHITECTURAL CONSTRAINTS FOR DIRECTML

### Safe Patterns
```python
# Causal buffer — torch.cat not in-place copy_
class CausalConv1d(nn.Conv1d):
    def forward(self, x):
        padded = torch.cat([self._pad_cache, x], dim=-1)
        self._pad_cache = x[..., -self.pad_len:].detach()
        return F.conv1d(padded, self.weight, self.bias,
                        self.stride, 0, self.dilation, self.groups)

# CPU-pinned layer with self-healing re-pin
class CpuPinnedConv(nn.Module):
    def __init__(self): super().__init__(); self.conv = nn.Conv1d(...).cpu()
    def forward(self, x):
        if next(self.conv.parameters()).device.type != 'cpu':
            self.conv = self.conv.cpu()
        return self.conv(x.cpu()).to(x.device)

# STFT — always CPU during training
def cpu_stft(x, n_fft, hop, window_cpu):
    sx = torch.stft(x.cpu(), n_fft=n_fft, hop_length=hop,
                    window=window_cpu, return_complex=True,
                    center=False, onesided=True)
    return sx  # keep on CPU or .to(device) as needed
```

### What to Avoid
- ConvTranspose1d with groups — broken backward
- Gradient through F.pad
- Gradient checkpointing
- GELU in any trainable layer

---

## 6. FASTAI-SPECIFIC PATCHES FOR DIRECTML

Apply patches after all model/loss/metric definitions (last notebook).

### Required Patches

**AvgSmoothLoss** — eliminate `aten::_foreach_lerp_`
```python
# Replace fastai's tensor EMA with pure Python float
# self._smooth = beta * self._smooth + (1-beta) * loss.item()
```

**SafeSaveModelCallback**
```python
# Guards IndexError on empty recorder
# Persists best metric to JSON sidecar → restored in before_fit on resume
```

**RecorderCleaner(order=80)**
```python
# Clears recorder.losses/lrs/iters after each epoch
# Prevents memory growth in 100+ epoch runs
```

**DMLQueueFlush(order=98)**
```python
# One .item() sync per epoch end
# Resets DML command queue depth accumulation
```

**_dml_safe_load_model**
```python
# Load to CPU first, then move to DML
state = torch.load(path, map_location='cpu', weights_only=False)
model.load_state_dict(state['model'])
model.to(dml_device)
# Handles Lookahead slow_weights correctly
```

### Training Resume Pattern
```python
# fastai start_epoch positions LR correctly via pct_train = epoch/n_epoch
# Always pass total epochs (not remaining)
learn.fit_one_cycle(
    total_epochs,          # total, never remaining
    lr_max=original_lr_max,
    div=original_div,
    div_final=original_div_final,
    pct_start=original_pct_start,
    wd=wd,
    start_epoch=epochs_done
)
# SkipToEpoch(order=70) raises CancelEpochException for skipped epochs
# ParamScheduler(order=60) reads pct_train=epoch/n_epoch → correct LR position
# Skipped epochs take negligible time (no forward/backward)
```

---

## 7. MICROSOFT DIRECTML EXAMPLES — KEY OBSERVATIONS

### Whisper Inference Example
- FP16 inference works (forward only)
- GELU safe in inference, broken in training backward
- torch.stft forward works on DML, backward does not
- `torch_directml.multi_head_attention` — inference-only fused kernel
- Uses integer mask not additive float causal mask

### Classification Training Example
- Microsoft uses SGD (safest optimizer for DML)
- `batch_loss.to('cpu')` before `.item()` — explicit CPU move before sync
- No mixed precision in their own training examples
- `loss.to(device)` on parameterless loss modules is a no-op, harmless

### General Rule
- Forward pass: most ops work on DML
- Backward pass: many ops broken — treat all backward ops as suspect until verified

---

## 8. ENVIRONMENT

```
GPU:       AMD Radeon RX 560X 4GB VRAM
Backend:   torch_directml (privateuseone:0)
Python:    3.12.9
Framework: fastai (with patches listed above)
Precision: FP32 only for training, FP16 acceptable for inference
OS:        Windows / Linux compatible
```
