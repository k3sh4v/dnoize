# Project Memory - CausalDNoizeConvTasNet Training
# Resume context for new chat. Paste this at the start of a new conversation.
# Also attach torch_directml_constraints.md for hardware constraint reference.

---

## PROJECT OVERVIEW

Training a causal audio denoiser (CausalDNoizeConvTasNet) on VoiceBank-DEMAND 28spk.
Goal: speech enhancement - remove noise and optionally dereverberate speech.
Hardware: AMD RX 560X 4GB GPU via torch_directml (privateuseone:0).
Framework: fastai, Python 3.12.9.

---

## DATASET

```
Source:        VoiceBank-DEMAND 28spk
Total pairs:   11,572 | train=9,837 | valid=1,735
valid_pct:     0.15
Sample rate:   16kHz
target_length: 40000 (2.5 seconds)
Batch shape:   [2, 1, 40000] noisy + [2, 1, 40000] clean
```

### Validation Split (CRITICAL – Use Fixed Seed)
- `RandomSplitter(valid_pct=0.15, seed=42)` **DO NOT use EPOCH_SEED.value**
- A varying seed changes validation composition every epoch -> metric noise
- Fixed seed ensures metric changes reflect model improvement, not data split variation

### dBFS Normalization (APPLIED - epoch 0 onward)
- Clean signal normalized to -18 dBFS (Podcast) independently before mixing
- Reverberant signal normalized to -18 dBFS independently
- Applied at get_x / get_y in fastai dataloader, not at batch level
- Do NOT normalize noisy+clean together (changes SNR)

### Augmentation Strategies
- denoise_only: add noise, no reverb
- joint: add noise + reverb (RT60=0.2-0.7s)
- deverb_only: add reverb only

---

## MODEL - CausalDNoizeConvTasNet

```
Parameters: ~216k
channels=64, num_blocks=11, num_repeats=2
HARD LIMIT: target_length=48000, channels=64, num_blocks=11, num_repeats=2 -> OOM. Stay at target_length=40000, channels=64, num_blocks=11, num_repeats=2

Architecture:
  encoder -> norm_enc (RMSNorm1d, device=dml) -> act_enc (ReLU) ->
  blocks (22x CausalDNoizeBlock) -> mask_head -> sigmoid ->
  masked (encoder_output * mask) -> decoder -> out_proj (CPU) ->
  trim/pad -> clamp(-1,1)*scale -> x + out

Key components:
  CausalConv1d:      torch.cat padding (never buf.detach()+copy_)
  CausalOutputProj:  on DML
  Gradient hook logs NaN when detected, guarded by requires_grad
  CausalDNoizeBlock: depthwise-separable + GLU + InstanceNorm(eps=1e-4), ReLU
  _init_weights:     xavier_uniform_ Conv1d, gain=0.01 mask_head/dec/out_proj
```

---

## Output scale curriculum (via shared multiprocessing value)

```
# In model forward():
OUTPUT_SCALE = multiprocessing.Value('d', 0.02)  # controlled by set_epoch_seed
return x + out.clamp(-1, 1) * OUTPUT_SCALE.value
```
# Phase 1 (epochs 1-200):  100% denoise,                         scale = 0.02
# Phase 2 (epochs 201-400): 50% denoise/40% deverb/10% joint,   scale ramp 0.02 -> 0.05
# Phase 3 (epochs 401-600): 25% denoise/15% deverb/60% joint,   scale ramp 0.05 -> 0.08

---

## NOTEBOOK RUN ORDER

```
helpers.ipynb -> callbacks.ipynb -> loss.ipynb -> metrics.ipynb ->
model.ipynb -> patches.ipynb (LAST)
```

---

## NUMERICAL STABILITY FIXES (APPLIED)
All these were necessary to eliminate NaN gradients:

1. Envelope loss epsilon placed INSIDE sqrt (not after)
```python
# BROKEN (produced SqrtBackward0 NaN):
p_env = p.pow(2).mean(-1).sqrt().clamp(min=1e-8)
# FIXED (finite gradient at zero):
p_env = torch.sqrt(p.pow(2).mean(-1) + 1e-4)
```
Root cause of the SqrtBackward0 runtime error you debugged.
Gradient of sqrt(0) = inf -> NaN. Adding epsilon inside keeps gradient finite.

2. Unified epsilon = 1e-4 across ALL loss and metric computations
```
All eps=1e-8 changed to 1e-4 throughout codebase
Components updated: si_snr_loss, SISNRMetric, SDRImprovement, SegSNRImprovement,
NoiseReductionPct, LogSpectralDistance, LSDOptimalGain, _envelope_loss,
MultiResolutionSpectralLoss, DMLSpectralLoss, helpers (normalize_to_dbfs,
_find_speech_crop_start, add_reverb), callbacks (SISNRDiagnostic, TrainingHealthCallback)
```

3. NaNGuard moved BEFORE GradientClip
```
NaNGuard.order = 4, GradientClip.order = 5
Previously NaNGuard was order 6 -> ran after clip, allowing NaN to spread
Now NaN is zeroed before clipping, preventing global norm corruption
```

4. Spectral loss gradient flows through .to() (no .detach())
```
MultiResolutionSpectralLoss computes on CPU, returns tensor on pred.device via .to()
The .detach() was removed – gradient now flows through the cross-device path
This is safe because the envelope loss epsilon fix prevents gradient explosion
```

## CURRENT LOSS

```python
# MultiResolutionSpectralLoss: CPU STFT, 3 resolutions [(256,64),(512,128),(1024,256)]
# Fully differentiable (gradient flows through .to(pred.device) without detach)
# _stft_losses per resolution:
mag_loss     = F.l1_loss(mag_x, mag_y)
log_mag_loss = F.l1_loss(torch.log(mag_x), torch.log(mag_y))
complex_loss = F.l1_loss(torch.view_as_real(sx), torch.view_as_real(sy))
return mag_loss + log_mag_loss + 0.5 * complex_loss

# CombinedLoss.forward (all on DML except spectral which is moved via .to()):
sisnr = si_snr_loss(pred_sq, targ_sq)
waveform = F.l1_loss(pred_sq.float(), targ_sq.float())
envelope = _envelope_loss(pred_sq.float(), targ_sq.float())  # uses sqrt(x + 1e-6)
tgrad = _temporal_gradient_loss(pred_sq.float(), targ_sq.float())
spec = self.spectral(pred, targ)  # CPU STFT, moved to DML via .to()
return sisnr + 0.40 * spec + 0.25 * waveform + 0.28 * envelope + 0.08 * tgrad
```

---

## CURRENT METRICS (updated - no external libraries)

```python
SISNRMetric             # sisnr_db, eps=1e-4
NoiseReductionPct       # noise_removed_% - denominator eps=1e-4
SDRImprovement          # sdr_improvement_db - eps=1e-4, SaveModelCallback PRIMARY monitor
SegSNRImprovement       # seg_snr_improvement_db - eps=1e-4
LogSpectralDistance     # log_spectral_dist - on CPU (torch.stft), then moved to DML
                        # Formula: sqrt(mean(log_diff^2, freq)), mean over frames
LSDOptimalGain          # lsd_optimal_gain - same CPU->DML pattern
```

### Metric Device Strategy (for STFT-based metrics LogSpectralDist/LSDOptimalGain):
Batches kept on DML
Per-sample slice moved to CPU only for torch.stft (required – complex64 unsupported on DML)
Magnitude result moved back to DML for remaining arithmetic
Scalar extracted with .item() – negligible overhead

---

## CURRENT OPTIMIZER

```python
# k=10 (increased from default 6 for more smoothing with augmented data)
# alpha=0.5 (standard Lookahead default)
def ranger_opt(params, **kwargs):
    kwargs.setdefault('decouple_wd', True)
    kwargs.setdefault('eps', 1e-7)
    return ranger(params, k=10, alpha=0.5, **kwargs)
```

---

## CURRENT TRAINING CONFIG (active run - 600 epochs)

```python
batch_size=2
GradientAccumulation(n_acc=2)   # effective batch = 4

training_lr_max = 1.4e-4
training_warmup = 0.1            # 60 epoch ramp, 540 epoch descent
training_div = 3.0               # lr starts at 1.4e-4/3 = 4.67e-5
training_div_final = 1400.0      # lr descends to 1.4e-4/1400 = 1e-7

learn.fit_one_cycle(
    600,
    lr_max=training_lr_max,
    div=training_div,
    div_final=training_div_final,
    pct_start=training_warmup,
    wd=1e-4,
    start_epoch=tracker.epochs_done
)
```

### Productive LR Zone (empirically validated)
Safe range: 1.0e-4 to 1.4e-4
lr_find result was too conservative (1e-6 to 5e-6) – curriculum + Lookahead introduce
gradient dynamics that short lr_find doesn't capture
Higher LRs (7.5e-4, 9e-3) triggered NaN gradients despite gradient clipping

---


## Callback Order (CRITICAL for gradient profiling)

```
NaNGuard                4   # zeros NaN gradients BEFORE clipping
GradientClip            5   # clips gradients (max_norm=3.0)
GradientProfileCallback 6   # captures POST-CLIP gradients in after_backward
TrainingHealthCallback  6   # captures POST-CLIP gradient stats in after_backward
GradientAccumulation    7   # n_acc=2, decides step vs accumulate
EpochTracker           60   # updates EPOCH_SEED, OUTPUT_SCALE
SafeSaveModelCallback  61   # saves best model
PeriodicPESQSTOI       70   # PESQ/STOI every N epochs
SISNRDiagnostic        82   # per-sample diagnostic
DMLQueueFlush          99   # epoch-end queue sync
```

---

## Gradient Capture Timing (CRITICAL)

- With n_acc=2, gradients are accumulated over 2 batches then zeroed on step
- NaNGuard (order 4) zeros NaN gradients before GradientClip (order 5)
- after_backward (order 6) fires AFTER clipping, BEFORE optimizer.step() and zero_grad()
- Profile/Health callbacks MUST use after_backward with order ≤ 7
- Capture on FIRST batch (batch_num=0) and LAST batch (batch_num=total-1) only to avoid 3x training slowdown from scanning all parameters every batch

---

## CURRICULUM AUGMENTATION (active - one-run curriculum)

```python
EPOCH_SEED = multiprocessing.Value('i', 0)      # current epoch number
OUTPUT_SCALE = multiprocessing.Value('d', 0.02)  # output scale multiplier

def set_epoch_seed(epoch, total_epochs):
    """Call before each epoch from main process"""
    EPOCH_SEED.value = epoch
    if epoch < 200:
        OUTPUT_SCALE.value = 0.02           # Phase 1
    elif epoch < 400:
        progress = (epoch - 200) / 200.0    # Phase 2: 0.02 -> 0.05
        OUTPUT_SCALE.value = 0.02 + 0.03 * progress
    else:
        progress = (epoch - 400) / 200.0    # Phase 3: 0.05 -> 0.08
        OUTPUT_SCALE.value = 0.05 + 0.03 * progress
```

### Deterministic Strategy Selection per File per Epoch
```python
def _get_strategy(path, epoch_seed, total_epochs=600):
    """
    600-epoch curriculum for denoise + deverb:
      Phase 1 (epochs   1-200): 100% denoise,                         rt60=(0.0, 0.0)
      Phase 2 (epochs 201-400):  50% denoise, 40% deverb,  10% joint, rt60=(0.1, 0.4)
      Phase 3 (epochs 401-600):  25% denoise, 15% deverb,  60% joint, rt60=(0.2, 0.7)
    """
```

---

## Phase transitions
- Phase 1->2 at epoch 200: loss will show a visible jump (task difficulty increases - expected)
- Phase 2->3 at epoch 400: same – not a training problem, the output scale also ramps
- EPOCH_SEED.value is used as epoch_seed arg for deterministic augmentation
- OUTPUT_SCALE.value adjusted per phase by set_epoch_seed

---

## PATCHES (patches.ipynb - all confirmed working)

1. `_dml_safe_load_model` - CPU load then DML move, sync optimizer state
2. `_patched_recorder_after_epoch` - ghost epoch guard
3. `AvgSmoothLoss` - pure Python float EMA, eliminates aten::_foreach_lerp_
4. `ProgressCallback` - handles Python float smooth_loss
5. `_dml_to_detach` - skips CPU move for privateuseone tensors
6. `SafeSaveModelCallback` - IndexError guard + JSON sidecar for best metric
   - before_fit auto-restores best from sidecar - no manual restoration needed
7. `RecorderCleaner(order=80)` - clears recorder after each epoch
8. `DMLQueueFlush(order=98)` - epoch-end queue flush

---

## EPOCH TRACKER (curriculum phase controller)

```python
_log_fields = [
  'epoch', 'train_loss', 'valid_loss',
  'sisnr_db', 'noise_removed_%',
  'sdr_improvement_db', 'seg_snr_improvement_db', 'log_spectral_dist',
  'lsd_optimal_gain', 'current_lr', 'lr_max', 'epoch_elapsed_time'
]
# Collects metric values by name (robust to metric order changes)
# metric_vals = {m.name: v for m, v in zip(learn.metrics, recorder.values[-1])}
```

---

## SISNR DIAGNOSTIC (updated - captures all metrics per fixed sample)

```python
"""
Per-sample diagnostic for fixed validation samples.
Captures SI-SNR, SDR, SegSNR, LSD for the same 2 samples each epoch.
Writes to training_stats/sisnr_diagnostic.csv.

Updated to be device-aware:
  - Batch tensors kept on DML
  - Per-sample slices moved to CPU only for torch.stft (if needed)
  - Remaining computation stays on DML
  - Scalar output via .item()
"""
_log_fields = [
  'epoch',
  'sisnr_sample_0', 'sisnr_sample_1',
  'sisnr_sample_2', 'sisnr_sample_3',
  'sisnr_mean',
  'sdr_sample_0', 'sdr_sample_1',
  'sdr_mean',
  'seg_snr_sample_0', 'seg_snr_sample_1',
  'seg_snr_mean',
  'lsd_sample_0', 'lsd_sample_1',
  'lsd_mean',
  'pred_min', 'pred_max',
  'pred_power_mean', 'targ_power_mean', 'power_ratio'
]
```

---

## METRICS UPDATE (device-aware STFT for LogSpectralDist and LSDOptimalGain)
```
Both spectral metrics now use keep the batch on DML and only move per-sample slices
to CPU for the mandatory torch.stft (DirectML does not support complex64).
The magnitude result is moved back to DML for log-difference and averaging.

All other metrics (SISNR, SDR, SegSNR) stay entirely on DML.
```

---
## GRADIENT HEALTH SUMMARY (empirical from epoch 1-40 training)
```
100 parameters, all non-zero and non-NaN throughout
grad_norm_mean varies from 0.001 to 1.8 (no concerning spikes)
grad_norm_max occasionally exceeds clip threshold (3.0) – expected early in training
power_ratio_db stable near 0 dB (output level correct)
No gradient anomaly detected in any epoch
```

---

## PeriodicPESQSTOI

```python
"""
Computes PESQ and STOI on CPU every N epochs on a small validation subset.
Never crashes training - all errors are caught and logged.

PESQ target: > 2.5
STOI target: > 0.88
"""
_log_fields = [
    'epoch',
    'pesq_mean', 'pesq_min', 'pesq_max',
    'stoi_mean', 'stoi_min', 'stoi_max',
    'n_samples', 'elapsed_seconds'
]
```

---

## GradientProfileCallback

```python
"""
  Profiles gradient health per named parameter every N epochs.
  Captures gradients AFTER backward (before optimizer zero_grad)
  on FIRST and LAST batch only — skips everything in between.
  Prints immediate warning on first batch if zero/NaN grads detected.
  Writes BOTH first and last snapshots to CSV for comparison.
  Works correctly with GradientAccumulation.
  Used to diagnose:
  - Which specific parameters have zero/NaN gradients
  - Whether zero grads are from SkipToEpoch (stale) or genuine
  - Whether a specific layer/module is consistently problematic
  - Gradient norm distribution across the model
"""
_log_fields = [
    'epoch',
    'batch_num',
    'batches_this_epoch',
    'capture_point',       # 'first', 'last', 'periodic'
    'param_name',
    'param_shape',
    'param_norm',
    'grad_norm',
    'grad_max',
    'grad_min',
    'is_zero',
    'is_nan',
    'is_inf',
    'grad_exists',
    'device',
]
# order = 6 (after GradientClip(5), before GradientAccumulation(7))
# capture_point: first=batch 0, last=batch total_batches-1
# First batch capture gives immediate warning if zero grads detected
# If zero_grads=0 and norm params on CPU: check param.device distribution
```

---

## TrainingHealthCallback

```python
"""
  Monitors training health every N epochs.
  Catches: frozen weights, gradient issues, optimizer staleness,
  EPOCH_SEED not advancing, model output collapse.
  Captures gradient stats in after_backward (first+last batch)
  (before optimizer zero_grad).
  Prints immediate warning on first batch if zero/NaN grads detected.
  Writes TWO rows per epoch (first and last) to CSV.
"""
_log_fields = [
    'epoch',
    'batch_num',
    'capture_point',
    'epoch_seed',
    'epoch_seed_advanced',
    'weights_changed',
    'max_weight_delta',
    'grad_norm_mean',
    'grad_norm_max',
    'nan_grads',
    'zero_grads',
    'output_rms',
    'target_rms',
    'power_ratio_db',
    'loss_value',
    'current_lr',
]
# order = 6 (after GradientClip(5), before GradientAccumulation(7))
# output_rms/target_rms/power_ratio_db: computed from one validation batch
#   - one forward pass per epoch, negligible overhead
#   - skip with skip_level_check=True if needed
```

---

## METRIC TARGETS (Phase 1, epochs 1-200, denoise only)

```
              Epoch 5    Epoch 50    Epoch 100   Epoch 200
sdr_impr_db   1.6        3.5-5.0     >5.0        >5.0
log_spec_dist 16.8       9-11        <8.0        <8.0
lsd_opt_gain  1.35       <1.5        <1.5        <1.5
sisnr_db      8.6        3-5         0-+5        0-+5
```

- LSDOptimalGain low + LogSpectralDist high = level mismatch (fixed by scale ramp in Phase 2)
- Gap between LSD and LSDOptimalGain will shrink as OUTPUT_SCALE increases

---

## DECISIONS MADE - DO NOT REVISIT

- FP16 training: NOT viable on DirectML
- Mel-scale weighting: NOT needed (log-magnitude loss approximates it)
- torch.utils.checkpoint: NOT usable on DirectML
- GELU: replaced with ReLU permanently
- AdamW: replaced with Adam(decouple_wd=True) permanently
- buf.detach()+copy_(): replaced with torch.cat permanently
- No sync/flush API in torch_directml: confirmed, no workaround exists
- PESQ/STOI replaced: SDRi + SegSNR + LSD are better fit (no external libs)
- dBFS normalization target: -18 dBFS, applied at dataloader level
- Validation split: fixed seed (42), NOT EPOCH_SEED.value
- Epsilon: unified to 1e-4
- register_buffer vs lru_cache: register_buffer correct for training windows
- wd in optimizer constructor: fastai overrides it - removed from constructor
- NaNGuard order: 4 (before GradientClip, order 5)
- Gradient path through spectral loss: enabled (no .detach())
- CausalOutputProj: on DML, hook guarded by requires_grad
- lr_find values too conservative – productive zone determined empirically: 1.0e-4 to 1.4e-4
- Output scale variation crucial: low scale in Phase 1 forces master denoising first

---

## WHAT TO DO IN NEXT CHAT

Priority order for next session:
1. Monitor phase transition at epoch 199 (strategy shift)
   - Expect visible loss jump and level mismatch reduction
2. Watch sdr_improvement_db as primary signal
3. Verify log_spectral_dist drops toward LSDOptimalGain as scale ramps
4. Consider if envelope loss weight (0.28) needs adjustment as deverb is introduced

---

## OPEN ITEMS

- [ ] Monitor Phase 2 transition at epoch 200 – check metrics don't regress
- [ ] Evaluate if higher LR (e.g., 2.0e-4) is viable after Phase 1 stabilizes
- [ ] Consider adding DNSMOS for modern perceptual evaluation (alternative to PESQ)
- [ ] No free open-source POLQA – use DNSMOS or NORESQA-MOS if needed

## COMPLETED

- [x] All DirectML broken ops identified and worked around
- [x] patches.ipynb stable and complete
- [x] Resume logic fixed (start_epoch + total epochs approach)
- [x] SafeSaveModelCallback with JSON sidecar - no manual restoration needed
- [x] Productive LR zone confirmed: 1.0e-4 to 1.4e-4
- [x] SqrtBackward0 NaN root cause identified and fixed (eps inside sqrt)
- [x] All epsilon values unified to 1e-4 across codebase
- [x] Spectral loss gradient re-enabled (no .detach() – safe after sqrt fix)
- [x] CausalOutputProj hook guarded against validation no_grad context
- [x] Device-aware metrics for LogSpectralDistance and LSDOptimalGain
- [x] Device-aware SISNRDiagnostic for STFT computation
- [x] Validation split now uses fixed seed (not EPOCH_SEED.value)
- [x] Output scale curriculum tied to curriculum phases
