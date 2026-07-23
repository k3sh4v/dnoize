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
target_length: 40000 (3 seconds)
Batch shape:   [2, 1, 40000] noisy + [2, 1, 40000] clean
```

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
  CausalOutputProj:  always on CPU, self-healing re-pin
  CausalDNoizeBlock: depthwise-separable + GLU + InstanceNorm(eps=1e-4), ReLU
  _init_weights:     xavier_uniform_ Conv1d, gain=0.01 mask_head/dec/out_proj
```

---

## Output scale curriculum (via shared multiprocessing value)

```
# In model forward():
OUTPUT_SCALE = multiprocessing.Value('d', 0.02)  # controlled by EpochTracker
return x + out.clamp(-1, 1) * OUTPUT_SCALE.value
```

---

## NOTEBOOK RUN ORDER

```
helpers.ipynb -> callbacks.ipynb -> loss.ipynb -> metrics.ipynb ->
model.ipynb -> patches.ipynb (LAST)
```

---

## CURRENT LOSS (updated - complex STFT + waveform L1 added)

```python
# MultiResolutionSpectralLoss: CPU STFT, 3 resolutions [(256,64),(512,128),(1024,256)]
# _stft_losses per resolution:
mag_loss     = F.l1_loss(mag_x, mag_y)
log_mag_loss = F.l1_loss(torch.log(mag_x), torch.log(mag_y))
complex_loss = F.l1_loss(torch.view_as_real(sx), torch.view_as_real(sy))
return mag_loss + log_mag_loss + 0.5 * complex_loss
# CombinedLoss.forward:
waveform = F.l1_loss(pred_sq.float(), targ_sq.float())
envelope = _envelope_loss(pred_sq.float(), targ_sq.float())
tgrad = _temporal_gradient_loss(pred_sq.float(), targ_sq.float())
return sisnr + 0.40 * _spec + 0.25 * waveform + 0.28 * envelope + 0.08 * tgrad
```

---

## CURRENT METRICS (updated - no external libraries)

```python
SISNRMetric             # sisnr_db
NoiseReductionPct       # noise_removed_% - (orig_noise_pwr - resid_noise_pwr)/orig * 100
SDRImprovement          # sdr_improvement_db - SaveModelCallback PRIMARY monitor
SegSNRImprovement       # seg_snr_improvement_db - proxy for CBAK
LogSpectralDistance     # log_spectral_dist - lower is better
                        # Formula: mean over freq per frame, then over time
```

---

## CURRENT OPTIMIZER

```python
# k=6->10 (more smoothing for augmented data)
# wd removed from constructor (fastai's fit call sets it via set_hypers)
def lookahead_adamw(params, lr=1e-3, eps=1e-7, **kwargs):
    return Lookahead(
        Adam(params, decouple_wd=True, eps=eps, **kwargs),
        k=10, alpha=0.5
    )
```

---

## CURRENT TRAINING CONFIG (active run - 300 epochs)

```python
batch_size=2
GradientAccumulation(n_acc=2)   # effective batch = 4

learn.fit_one_cycle(
    600,
    lr_max=9.0e-5,
    div=3.0,
    div_final=900.0,
    pct_start=0.1,    # 60 epoch ramp, 540 epoch descent
    wd=1e-4,
    start_epoch=tracker.epochs_done
)
# LR schedule:
# Epochs   1- 60: cosine ramp 3.0e-5 -> 9.0e-5
# Epochs  60-600: cosine descent 9.0e-5 -> 1.0e-7
```

---

## GradientClip(max_norm=3.0)

```
GradientClip(max_norm=3.0)
```

---

## Callback Order (CRITICAL for gradient profiling)

```
TrainEvalCallback     -10
Recorder               50
CastToTensor            9
ProgressCallback       60
SafeSaveModelCallback  61
GradientClip            5   # clips gradients
GradientProfileCallback 6   # captures POST-CLIP gradients in after_backward
TrainingHealthCallback  6   # captures POST-CLIP gradient stats in after_backward
GradientAccumulation    7   # n_acc=2, decides step vs accumulate
SISNRDiagnostic        82
PeriodicPESQSTOI       70
DMLQueueFlush          99
EpochTracker           60   # updates EPOCH_SEED, OUTPUT_SCALE, EPOCH_RT60
```

---

## Gradient Capture Timing (CRITICAL)

- With n_acc=2, gradients are accumulated over 2 batches then zeroed on step
- after_backward (order 6) fires BEFORE optimizer.step() and zero_grad()
- Capturing in after_epoch shows ALL zeros because zero_grad already ran
- Profile/Health callbacks MUST use after_backward with order ≤ 6
- Capture on FIRST batch (batch_num=0) and LAST batch (batch_num=total-1) only
- to avoid 3x training slowdown from scanning all 100 parameters every batch
- First batch capture gives immediate warning if zero grads detected at epoch start

---

## CURRICULUM AUGMENTATION (active - one-run curriculum)

```python
EPOCH_RT60 = multiprocessing.Value('d', 0.0)      # controlled by EpochTracker
OUTPUT_SCALE = multiprocessing.Value('d', 0.02)     # controlled by EpochTracker

def _get_strategy(path, epoch_seed, total_epochs=600):
    """
    Deterministic curriculum strategy selection per file per epoch.
    Same path + epoch -> same strategy always.

    600-epoch curriculum for denoise + deverb:
      Phase 1 (epochs   1-200): 100% denoise,                         rt60=(0.0, 0.0)
      Phase 2 (epochs 201-400):  50% denoise, 40% deverb,  10% joint, rt60=(0.1, 0.4)
      Phase 3 (epochs 401-600):  25% denoise, 15% deverb,  60% joint, rt60=(0.2, 0.7)

    Returns: (strategy, rt60_range)
      strategy:   'denoise_only' | 'deverb_only' | 'joint'
      rt60_range: (min_rt60, max_rt60) for reverb augmentation
    """
    epoch = int(epoch_seed)
    h = abs(hash((str(path), epoch))) % 1000
    if epoch < 200:
        # Phase 1 - pure denoise foundation
        return 'denoise_only', (0.0, 0.0)
    elif epoch < 400:
        # Phase 2 - introduce deverb, maintain denoise to prevent forgetting
        if h < 500:       # 50%
            return 'denoise_only', (0.0, 0.0)
        elif h < 900:     # 40%
            return 'deverb_only', (0.1, 0.4)
        else:             # 10%
            return 'joint', (0.1, 0.4)
    else:
        # Phase 3 - emphasis on joint, still maintain all skills
        if h < 250:       # 25%
            return 'denoise_only', (0.0, 0.0)
        elif h < 400:     # 15%
            return 'deverb_only', (0.2, 0.7)
        else:             # 60%
            return 'joint', (0.2, 0.7)
```

---

## Phase transitions
- Phase 1->2 at epoch 199: loss will show a visible jump (task difficulty increases - expected)
- Phase 2->3 at epoch 399: same - not a training problem
- EPOCH_SEED (shared multiprocessing Value) must be passed as epoch_seed arg
- EPOCH_RT60 updated by EpochTracker to reflect current phase rt60 floor
- OUTPUT_SCALE updated by EpochTracker to reflect current phase output scale

---

## PATCHES (patches.ipynb - all confirmed working)

1. `_dml_safe_load_model` - CPU load then DML move, handles Lookahead slow_weights
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
    Per-sample SI-SNR diagnostic for fixed validation samples.
    Also captures pred power, target power, power ratio, and all
    metrics (SDRi, SegSNR, LSD) for the same fixed samples.
    Writes to training_stats/sisnr_diagnostic.csv.
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
# Fixed batch captured from first validation batch in before_validate
# after_validate computes all metrics per sample and writes row
# Deeply negative SISNR on a sample (-13 to -30 dB) indicates reverb failure
# Positive SISNR on all samples = model is healthy for that sample type
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

## DECISIONS MADE - DO NOT REVISIT

- FP16 training: NOT viable on DirectML
- Mel-scale weighting: NOT needed (log-magnitude loss approximates it)
- torch.utils.checkpoint: NOT usable on DirectML
- num_blocks=11: OOM - stay at 10
- GELU: replaced with ReLU permanently
- AdamW: replaced with Adam(decouple_wd=True) permanently
- buf.detach()+copy_(): replaced with torch.cat permanently
- No sync/flush API in torch_directml: confirmed, no workaround exists
- register_buffer vs lru_cache: register_buffer correct for training windows
- wd in optimizer constructor: fastai overrides it - removed from constructor
- fit_flat_cos = fit_one_cycle(div=1): confirmed equivalent from source
- PESQ/STOI replaced: SDRi + SegSNR + LSD are better fit (no external libs)
- dBFS normalization target: -23 dBFS (EBU R128), applied at dataloader level
- Curriculum mix final phase: 40% denoise / 35% joint / 25% deverb

---

## WHAT TO DO IN NEXT CHAT

Priority order for next session:
1. Monitor phase transition at epoch 199 (strategy shift)
   - expect visible loss jump, not a problem
2. Watch sdr_improvement_db as primary signal
3. Verify log_spectral_dist trend
4. Consider fixed held-out eval set (no augmentation) for clean PESQ comparison

---

## OPEN ITEMS

- [ ] Fix SISNRDiagnostic sisnr_sample_2/3 always empty (data loading bug)
- [ ] Implement RIR cache (rir_cache.ipynb - design ready, not implemented)
- [ ] Add fixed held-out eval set (no augmentation) for cross-run PESQ comparison
- [ ] Consider envelope consistency loss (multi-scale RMS) - next-next run

## COMPLETED

- [x] All DirectML broken ops identified and worked around
- [x] patches.ipynb stable and complete
- [x] Resume logic fixed (start_epoch + total epochs approach)
- [x] SafeSaveModelCallback with JSON sidecar - no manual restoration needed
- [x] Productive LR zone confirmed: 1.0e-4 to 1.4e-4
- [x] Phase-blind loss identified -> fixed with complex STFT + waveform L1
- [x] dBFS normalization at -23 dBFS applied at dataloader level
- [x] Curriculum augmentation implemented and active
- [x] Optimizer updated: k=10, wd removed from constructor
- [x] Metrics replaced: SDRi + SegSNR + LSD (no external libraries)
- [x] SaveModelCallback monitor changed to sdr_improvement_db
- [x] EpochTracker updated to capture all 7 metrics by name
- [x] SISNRDiagnostic updated to capture SDRi + SegSNR + LSD per fixed sample
- [x] LogSpectralDistance formula fixed (mean over freq first, then time)
- [x] fit_one_cycle configured for productive LR zone (1.19e-4 to 1.40e-4)
- [x] Whisper + classification DirectML examples fully analyzed
- [x] fastai fit_one_cycle/SkipToEpoch/ParamScheduler source confirmed
