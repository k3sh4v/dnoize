# Audio Task Knowledge Decisions
## CausalDNoizeConvTasNet - Speech Denoising and Dereverberation

This document captures every audio-domain decision made during the design and training of this
model - what was chosen, why it was chosen, and how it contributes to building a high-quality
causal speech enhancement system. It is intended as a reference for future runs, extensions to
music, and onboarding anyone working on this system.

---

## 1. Task Definition and Why It Is Hard

Speech enhancement combines two distinct subtasks. Denoising removes additive noise - broadband
signals like office noise, traffic, or crowd sounds that are statistically independent from speech.
Dereverberation removes room reflections - time-delayed, amplitude-decayed copies of the speech
signal itself, caused by sound bouncing off walls, floors, and ceilings before reaching the
microphone.

These tasks are hard for different reasons. Noise is statistically separable from speech because
it has a different spectral profile - the model can learn to identify which frequency bins contain
primarily noise energy and suppress them. Reverberation is much harder because it is derived from
the speech signal itself. The reflected energy has the same spectral structure as the direct signal,
just delayed and attenuated. The model must learn the temporal causality of speech - that energy
arriving slightly after a phoneme boundary is reflected, not direct - which requires longer
temporal context and more refined representations.

This is why the curriculum trains denoising first. The model learns the noise-speech boundary
before being asked to distinguish direct from reflected speech. Mixing both tasks from the start
causes the model to learn a compromise that is mediocre at both.

---

## 2. Architecture Decisions

### 2.1 CausalConvTasNet - Why This Architecture

Conv-TasNet was chosen because it is a proven architecture for audio source separation that
operates directly on waveforms without requiring an intermediate spectrogram. The encoder learns
a learned filterbank that is more flexible than a fixed STFT, and the mask-based approach
(predict a mask over the encoder representation, not the output directly) has strong theoretical
justification - it forces the model to learn what to keep rather than what to add.

The causal variant is required for real-time inference. A non-causal model looks at future samples
to inform its current output. This is fine for offline processing but impossible for streaming
applications where the future has not arrived yet. Making the model causal means all convolutions
only look backward in time, enabling frame-by-frame real-time processing.

### 2.2 Depth vs Width - Why num_blocks Matters More Than channels

This is one of the most important architecture decisions in this system. Depth (num_blocks) and
width (channels) both increase model capacity, but they increase different kinds of capacity.

Width (channels) increases the dimensionality of the feature representation at each timestep.
More channels means the model can represent more complex spectral patterns simultaneously - richer
frequency decompositions, more nuanced noise profiles. But all of this representation happens
locally at each time position.

Depth (num_blocks with exponentially increasing dilation) increases the temporal receptive field.
Each block at dilation d can see 2d samples into the past. With 11 blocks and dilation doubling
each block (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), the maximum dilation is 1024. With
num_repeats=2 this doubles to 2048. At 16kHz the effective receptive field spans 128ms of context
per block, and 256ms total across the stack.

For speech enhancement, temporal context is more valuable than spectral richness. Why:

- Noise suppression requires understanding that a burst of energy is noise, not a consonant.
  This requires comparing energy patterns over time, not just at a single frame.
- Dereverberation requires understanding that energy arriving after a phoneme onset is a
  reflection. The RT60 of a room is 0.2-0.7 seconds - 3200 to 11200 samples at 16kHz. Only
  a model with sufficient receptive field can see far enough back to model this correctly.
- Speech has prosodic structure (stress patterns, phrase boundaries) that spans 500ms or more.
  A model that cannot see this context will make different suppression decisions at stressed vs
  unstressed syllables, creating audible quality variations.

The ablation test of channels=80, num_blocks=8 vs channels=64, num_blocks=10 confirmed this
principle empirically. Despite having more parameters in the channel dimension, the wider-shallower
model performed worse at epoch 3 on every metric and ran 35% slower per epoch. Depth won.

The max dilation for num_blocks=8 is 2^7=128 vs 2^9=512 for num_blocks=10. At kernel_size=3,
the receptive field difference is roughly 384 samples (24ms) vs 1536 samples (96ms). For a
reverb tail lasting 200-600ms, 24ms of context is simply not enough.

Rule for future architecture experiments: increase num_blocks before increasing channels.
If VRAM allows, try channels=64, num_blocks=12 before channels=80, num_blocks=10.

### 2.3 RMSNorm1d After Encoder - Why It Matters (Updated from InstanceNorm)

RMSNorm1d replaced InstanceNorm1d as the normalization layer after testing revealed it provides
better gradient flow on DirectML hardware while maintaining the same amplitude-invariance property.

RMSNorm normalizes each channel's activations to unit RMS (root-mean-square) per sample per
channel. This has two important effects:

First, it makes the model's internal representations amplitude-invariant. Regardless of whether
the input is loud or quiet, the feature maps entering the mask prediction blocks are normalized.
This means the mask learning task is decoupled from the amplitude estimation task. Without this,
the model would waste capacity learning that loud inputs need bigger corrections than quiet ones.

Second, it makes the model robust to the dBFS normalization target. When the inference pipeline
normalizes input to -18 dBFS, RMSNorm ensures the internal processing is unaffected by any
small normalization inaccuracies. Only the final residual connection (x + output) sees the raw
amplitude, and that only needs to adjust a small output projection layer.

**Critical implementation detail:** RMSNorm1d weight and bias must be created on the SAME device
as the model, not defaulting to CPU. The standard `nn.Parameter(torch.ones(1, C, 1))` creates
parameters on CPU even when the model is moved to DML via `.to(dml_device)`. This causes a
cross-device autograd break where Conv1d gradients become exactly zero while norm gradients
remain non-zero. Always use `torch.ones(1, C, 1, device=device)` and pass the device through the
constructor.

### 2.4 GLU (Gated Linear Units) - Selective Information Flow

Each DNoize block uses a Gated Linear Unit implemented as split + ReLU + multiply. GLU is a
learned gating mechanism - one half of the channel dimension acts as a gate that controls how
much of the other half passes through. This is directly analogous to how LSTM gates work, applied
to 1D convolutions.

For speech enhancement, GLU is particularly valuable because speech has highly non-stationary
structure. Voiced segments (vowels) have very different spectral characteristics from unvoiced
segments (fricatives) and silence. GLU allows the model to selectively propagate different feature
patterns depending on what type of speech is currently being processed. A model without gating
applies the same linear transformation everywhere, which is less flexible.

ReLU was chosen over GELU because GELU backward is broken on DirectML. This is a hardware
constraint but happens to be a reasonable choice anyway - ReLU gates are binary (a feature either
passes or it doesn't) while GELU is a smoother approximation. For a gating mechanism, the sharper
ReLU boundary is often preferable.

### 2.5 Stride-2 Encoder and Decoder Reconstruction

The encoder uses a stride of 2, halving the temporal resolution from 48000 to 24000 samples. The
decoder must upsample back to the original length. This is a standard pattern in Conv-TasNet
and related architectures.

For noise removal, stride-2 is sufficient because noise suppression operates primarily in the
frequency/spectral domain, where halving temporal resolution does not significantly impair the
model's ability to identify and suppress noise components. The mask operates on the encoded
representation at half resolution, and the decoder reconstructs the full-resolution signal.

For dereverberation, stride-2 creates a challenge because reverb has fine temporal structure
(early reflections, pre-delay, decay tail) that may be partially lost at half resolution. The
decoder must simultaneously reconstruct the missing temporal samples AND remove reverberation.
This dual burden means the decoder layer is a potential bottleneck for the dereverberation task.

The stride-2 design was chosen because it reduces computational cost by approximately 2x in the
blocks (which operate at the lower resolution), enabling deeper networks within the same VRAM
budget. This tradeoff is acceptable for denoising but may limit dereverberation performance. If
VRAM allows, a stride-1 encoder (full resolution throughout) would improve dereverberation at
the cost of halving network depth or doubling training time.

### 2.6 Residual Connection with Scaled Output

The final output is `x + clamp(out, -1, 1) * 0.02` where x is the input and out is the model's
correction. This residual formulation means the model learns to predict the correction to apply to
the noisy input rather than predicting the clean signal from scratch.

This is much easier to learn. The clean signal is correlated with the noisy input (they share the
same speech content). A model that starts from the noisy input and makes small corrections can
leverage this correlation from the first epoch. A model that must predict the clean signal from
scratch has no such shortcut.

The 0.02 scaling factor constrains the initial output magnitude, preventing the model from making
large destructive corrections early in training when the gradients are noisy. As training
progresses, the model learns to compensate for this scaling through the weight magnitudes in
out_proj.

---

## 3. Loss Function Decisions

### 3.1 Why SI-SNR Is the Primary Loss

Scale-Invariant Signal-to-Noise Ratio projects the predicted signal onto the target and measures
the ratio of signal power to residual power. The scale-invariant property means it ignores absolute
amplitude - a prediction that is twice as loud but otherwise identical to the target scores the
same as a perfect match. This is correct because the model's job is to produce the right signal
shape, not the right amplitude (amplitude is handled by the normalization pipeline).

SI-SNR is the dominant term (weight 1.0) because it directly measures what we care about:
how much of the predicted signal is target speech vs residual noise. It is sensitive to phase
because a phase-shifted prediction will project poorly onto the target direction, giving low SI-SNR.
It is sensitive to temporal alignment because a time-shifted signal also projects poorly.

SI-SNR as a loss is negated (higher SI-SNR = lower loss) which is why valid_loss is negative in
the training logs. A valid_loss of -5.5 means the model is achieving approximately +5.5 dB SI-SNR
improvement over the noisy input.

### 3.2 Why Multi-Resolution Spectral Loss Supplements SI-SNR

SI-SNR operates on the entire waveform at once. It gives equal weight to every sample in the 3-second
window. Speech has very non-uniform perceptual importance - a 20ms consonant burst matters far more
for intelligibility than the same energy in a steady vowel.

Multi-resolution STFT loss operates on different time-frequency representations simultaneously.
The three resolutions chosen are:

- 256-point FFT (16ms window): captures consonants, fricatives, stop bursts - fast transients
- 512-point FFT (32ms window): captures vowel formants, pitch periods - mid-timescale structure
- 1024-point FFT (64ms window): captures pitch, prosody, fundamental frequency - slow structure

Together they provide supervision at the timescales that matter for speech intelligibility and
naturalness. No single resolution can see all of these simultaneously - a long window smears
transients, a short window misses pitch structure.

### 3.3 Why Complex STFT, Not Just Magnitude

The original spectral loss only used magnitude - it compared |STFT(pred)| to |STFT(clean)|.
This is phase-blind. A predicted signal with correct magnitude but wrong phase sounds exactly
like the correct signal to the magnitude loss but sounds terrible to a human listener - it
produces comb filtering, hollow resonances, and unnatural timbre.

Adding complex L1 loss - comparing the real and imaginary components of the STFT directly -
penalizes phase errors. A prediction with correct magnitude but 90-degree phase shift has
zero magnitude error but large complex error. This forces the model to learn correct phase
relationships, which is essential for natural-sounding speech reconstruction.

The complex loss is weighted at 0.5 relative to mag + log_mag within each resolution to prevent
it from dominating. Phase errors are harder to optimize than magnitude errors and too much weight
on complex loss early in training can destabilize convergence.

### 3.4 Why Log-Magnitude in Addition to Linear Magnitude

Human hearing is logarithmic in both frequency and amplitude. A 1 dB error at 20 dB SPL sounds
as noticeable as a 1 dB error at 80 dB SPL. Linear magnitude loss treats a 0.01 error on a bin
with magnitude 0.01 (100% relative error) the same as a 0.01 error on a bin with magnitude 1.0
(1% relative error). Log-magnitude loss corrects this - it gives equal weight to equal relative
errors regardless of absolute magnitude.

For speech enhancement this matters because the model must suppress noise in low-energy frequency
bands without damaging the low-energy upper harmonics of speech. A linear loss would ignore these
low-energy details. Log-magnitude loss ensures they receive appropriate gradient signal.

### 3.5 Why Waveform L1 Is Included

SI-SNR and spectral losses operate on either global projections or frequency-domain representations.
Neither directly penalizes sample-level waveform accuracy. Waveform L1 - comparing pred[t] to
clean[t] for every sample - provides a direct gradient signal that the other terms lack.

Its weight (0.25) makes it meaningful but not dominant. At the estimated contribution of ~0.5%
of total loss magnitude, it provides a stabilizing signal without competing with the more
informative SI-SNR and spectral terms. It is particularly useful early in training when the model
is making large errors - the L1 gradient is constant regardless of error magnitude, unlike L2
which diminishes as errors shrink.

### 3.6 Why Envelope Loss - Temporal Structure

Reverb specifically damages the temporal envelope - the way speech energy rises and falls over
time. A reverberant signal has the same average spectral content as a clean signal but the energy
is smeared forward in time. The sharp attack of a consonant is followed by a decaying tail of
reflected energy, making the speech sound blurred and muddy.

SI-SNR, spectral loss, and waveform L1 all compare the model output to the target but none of
them specifically tell the model "get the energy timing right." A model could achieve reasonable
SI-SNR by getting the spectral content correct while having the energy arrive at slightly wrong
times, which sounds like mild reverb.

Envelope loss computes the RMS energy in short windows (4ms, 16ms, 64ms) and compares them in
log domain. Log domain is critical - it matches how humans perceive loudness variations and ensures
the loss treats a relative envelope error the same way regardless of absolute level. Three window
sizes capture different temporal scales: 4ms catches consonant onsets, 16ms catches syllable
nuclei, 64ms catches prosodic stress patterns.

This term becomes increasingly important as the curriculum introduces dereverberation at epoch 200.
The model's ability to restore correct energy timing is directly what envelope loss trains.

### 3.7 Why Temporal Gradient Loss - Transient Artifacts

Neural audio models commonly produce "musical noise" - isolated spectral artifacts that appear
and disappear rapidly, creating a metallic or bubbling quality. This happens because the mask
predictor occasionally produces inconsistent masks for adjacent time frames, causing rapid
amplitude changes in individual frequency bins.

Temporal gradient loss penalizes the first-order temporal derivative of the waveform: it compares
(pred[t+1] - pred[t]) to (clean[t+1] - clean[t]) for every consecutive sample pair. A correct
rapid change (a consonant onset) has the same gradient in both pred and clean and scores well. An
artifact (a spurious spike) appears in pred but not clean and is penalized.

Its weight (0.08) is small because excessive temporal gradient loss would over-smooth consonants,
making speech sound muffled. The right balance is enough weight to suppress artifacts without
damaging intentional transients.

### 3.8 Why power_reg Was Removed

Power regularization penalized the log ratio of predicted RMS to target RMS. It was useful early
in training to prevent the model from collapsing to silence (a trivially quiet output has zero
noise but also zero speech). Once the model is past the initial instability phase, SI-SNR already
handles scale via its projection mechanism, and the dBFS normalization ensures both input and
target are at consistent levels. At epoch 49, power_reg's estimated contribution was 0.008% of
total loss magnitude - too small to influence gradient direction. Keeping it added computational
overhead without providing signal. Removed.

### 3.9 The Loss Priority Hierarchy and Why

The ordering spec > envelope > waveform > tgrad reflects the information hierarchy of speech
enhancement at the training stage where these weights were chosen:

Spectral loss (weight 0.40) is highest among secondary terms because frequency-domain errors
are the most perceptually salient. A model that gets the waveform approximately right but has
spectral coloration will sound unnatural even if SI-SNR is high.

Envelope loss (weight 0.18) ranks second because temporal energy distribution is what
distinguishes dereverberation success from failure. The deverb curriculum makes this increasingly
important from epoch 199 onward.

Waveform L1 (weight 0.25 in nominal scale, but lower actual contribution) provides sample-level
accuracy as a baseline constraint. It ensures the model cannot achieve good spectral and envelope
metrics through a solution that is globally correct but locally incoherent.

Temporal gradient (weight 0.08) is last because it addresses artifacts - a failure mode that
only becomes relevant once the model has learned the primary task reasonably well. Early in
training penalizing temporal inconsistency is less important than penalizing spectral error.

---

## 4. Training Strategy Decisions

### 4.1 Curriculum Augmentation - Why Three 200-Epoch Phases

The three-phase curriculum (200 epochs each, 600 total) reflects the difficulty hierarchy of the
three tasks and prevents catastrophic forgetting.

**Phase 1 (epochs 1-200): Pure denoise foundation**
- 100% denoise_only, rt60=(0.0, 0.0), scale=0.02
- Model learns noise-speech boundary without reverb complication
- Expected outcome: SDR improvement 4-6 dB by epoch 200

**Phase 2 (epochs 201-400): Introduce dereverberation while maintaining denoise**
- 50% denoise_only (maintain skill, prevent forgetting)
- 40% deverb_only (learn reverb removal on clean signal)
- 10% joint (early exposure to combined task)
- rt60 ramps from (0.1, 0.4), scale ramps from 0.02 to 0.05
- Expected outcome: SDR improvement 5-7 dB, reverb sample SISNR turns positive

**Phase 3 (epochs 401-600): Full difficulty with maintenance**
- 25% denoise_only (maintenance)
- 15% deverb_only (maintenance)
- 60% joint (primary production task)
- rt60=(0.2, 0.7), scale=0.05 to 0.08
- Expected outcome: SDR improvement 6-10 dB, all sample types positive SISNR

The 200-epoch phase duration was chosen because:
- 200 epochs at current rate (2.5h/epoch) = ~500 hours per phase
- This is enough time for the model to converge on each subtask before difficulty increases
- Shorter phases (100 epochs) caused the model to hit difficulty jumps before converging on the
  previous task, resulting in loss spikes and metric instability

**Critical principle: never train on just one task after the other.** Pure deverb-only for 200
epochs would cause the model to forget denoise entirely, requiring relearning from scratch in
Phase 3. The 50% denoise maintenance in Phase 2 preserves the learned skill.

Deterministic strategy selection (hash of path + epoch) ensures the same file always gets the
same treatment in a given epoch. This means validation metrics are comparable across epochs -
the same validation files face the same augmentation conditions, so metric changes reflect model
improvement, not data variation.

### 4.2 Why Not Fine-Tune or Two-Model

Fine-tuning (train 300 epochs denoise, then fine-tune on reverb) was rejected because it shares
the same catastrophic forgetting problem as pure sequential training. The only way to prevent
forgetting is to include denoise samples during the reverb phase, which is exactly what the
curriculum does.

A two-model approach (separate denoise and deverb models with detection/routing) was considered
but rejected because:
- Joint processing exploits shared representations between noise and reverb
- Reverb detection is unreliable in practice
- Two models in series doubles inference latency
- The interaction between noise and reverb means sequential processing is suboptimal
- A single 216k-parameter model fits easily in 4GB VRAM; two models is unnecessary complexity

### 4.3 dBFS Normalization at -18 - Why This Level

Audio normalization before the model serves two purposes. First, it prevents the model from
wasting capacity learning amplitude compensation - without normalization, the model must learn
that a quiet input needs less absolute correction than a loud one, which is trivially handled by
normalization. Second, it ensures the loss functions operate in a consistent amplitude regime.

-18 dBFS (AES streaming standard) was chosen over -23 dBFS (EBU R128 broadcast) for one practical
reason: the model is intended for real microphone speech, which typically arrives at -20 to -6
dBFS depending on microphone gain and speaker distance. -18 dBFS is closer to this practical
range, meaning the inference normalization (normalize input to -18, process, restore level) applies
a smaller gain factor to typical inputs. Smaller normalization factors mean less amplification of
any pre-normalization noise floor.

The normalization is applied to the clean signal and the reverberant/noisy signal independently,
not jointly. Normalizing jointly would change the SNR ratio - if a quiet clean signal is mixed
with loud noise, joint normalization would make the noise appear relatively louder. Independent
normalization preserves the noise type and level characteristics that the model is trained to
handle.

Normalization is kept in the preprocessing pipeline rather than trained into the model for one
reason: it adds less than a microsecond of latency and requires no learned parameters. Training
the model to learn normalization would consume capacity that is better spent on the actual
enhancement task.

### 4.4 Learning Rate Schedule - Why the Productive Zone Matters

The one-cycle learning rate schedule (fit_one_cycle) is used because it has three distinct phases
that serve different purposes.

The warmup phase (pct_start=0.1, so 60 epochs out of 600) ramps LR from lr_max/div to lr_max.
During warmup the model is exploring - high LR allows large weight updates that can escape poor
initial configurations. The gradients are noisy and the loss landscape is rough, so the model
benefits from a cautious start.

The descent phase (remaining 540 epochs) gradually reduces LR from lr_max to lr_max/div_final.
As the model converges toward a good solution, smaller LR steps are needed to refine it without
overshooting. The cosine shape (slow at start and end, fast in middle) matches the empirically
observed learning dynamics - large improvements when escaping poor configurations, diminishing
returns as the model approaches convergence.

### 4.5 Gradient Accumulation - Why Effective Batch 4

With batch_size=2 and n_acc=2, the optimizer sees the gradient from 4 samples before each
weight update. The effective batch size of 4 was chosen because:

- Batch size 2 is the maximum that fits in VRAM at 40000 samples with 64 channels and 22 blocks
- n_acc=2 doubles the effective batch to 4, providing slightly more stable gradient estimates
- With 9837 training samples and effective batch 4, each epoch contains 2459 optimizer steps
- This is fewer steps than ideal (would prefer ~400-500) but acceptable given VRAM constraints

The original plan was n_acc=12 (effective batch 24), but this was found to be unnecessary and
caused callback timing issues with gradient profiling. n_acc=2 is simpler and sufficient.

### 4.6 RIR Pool - Why Pre-Generation

Generating Room Impulse Responses on-the-fly using image method simulation is CPU-intensive.
Each RIR generation involves computing reflection paths from source to microphone via multiple
wall reflections - a simulation that can take 10-50ms per sample. With 4918 training samples
per epoch and strategy-dependent augmentation, this adds meaningful overhead to every epoch.

Pre-generating a pool of 1735 RIRs at startup (one per validation sample in the dataset) and
indexing into the pool deterministically using hash(path, epoch) achieves the same augmentation
diversity at zero per-sample generation cost. The pool is fixed with a seed, ensuring it is
identical across restarts - critical for comparable metrics across resumed training runs.

1735 pool entries were chosen because it equals valid_pct × total_files. This gives enough
diversity that the model sees many different room conditions without memorizing specific RIRs,
while keeping RAM usage to approximately 55-88 MB.

### 4.7 Gradient Profiling and Callback Timing

Gradient profiles must capture in `after_backward` (order ≤ 6), NOT `after_epoch` or `after_batch`.
With GradientAccumulation(n_acc=2), the optimizer calls `zero_grad()` after every 2nd batch.
By `after_epoch`, all DML parameter gradients have been zeroed, so profiles would show all zeros.

Correct callback order:
```
NaNGuard                order=4   (before GradientClip)
GradientClip            order=5   (clips if norm > 3.0)
GradientProfileCallback order=6   (captures post-clip gradients)
TrainingHealthCallback  order=6   (captures post-clip gradient stats)
GradientAccumulation    order=7   (decides step vs accumulate)
DMLQueueFlush           order=99  (forces sync after epoch)
```
Capture on FIRST batch (batch 0) and LAST batch (total-1) only to avoid 3x training slowdown
from scanning all 100 parameters on every batch. First batch gives immediate zero-grad warning.
Last batch shows epoch-end state.

---

## 5. Metric Decisions

### 5.1 SDR Improvement (Primary Monitor)

Signal-to-Distortion Ratio improvement over the noisy input is the primary SaveModelCallback
monitor because it is a relative metric that directly answers the question "is the model output
better than doing nothing?" An SDRi of 0 dB means the model adds as much distortion as it
removes. Positive SDRi means genuine improvement.

SDR is computed as: 10 × log10(||target||² / ||pred - target||²). The error term (pred - target)²
is computed on the raw waveform, making SDRi naturally phase-sensitive - a phase-shifted output
has large waveform error even with correct magnitude. This is the right property for a quality
monitor.

SDRi is preferred over raw SI-SNR as the save monitor because it is relative - it compares the
model output to the noisy input, not to some absolute scale. This means a new best SDRi
checkpoint always represents genuine improvement over the noisy input baseline, regardless of
the absolute level of the training metrics.

### 5.2 Segmental SNR Improvement

SegSNR computes SNR in 20ms frames rather than over the entire signal. The 20ms frame size (320
samples at 16kHz) corresponds approximately to the duration of a single phone - the smallest
meaningful unit of speech. Computing SNR per phone rather than globally prevents the metric from
being dominated by long voiced vowels and ignoring short consonant bursts.

SegSNR improvement (SegSNRi) correlates well with CBAK (background noise quality), one of the
standard VoiceBank-DEMAND composite scores. This makes it a useful proxy for what would otherwise
require the external PESQ/STOI tools. Higher SegSNRi means the model consistently improves
noise suppression across all speech sounds, not just the loudest segments.

### 5.3 Log Spectral Distance

LSD measures the average spectral shape distortion in dB - specifically the RMS difference in
log power spectrum between pred and clean, averaged per frame and then over time. Unlike SI-SNR
and SDR which measure overall waveform quality, LSD is specifically sensitive to spectral
coloration: does the enhanced speech have the right formant structure, the right vowel quality,
the right consonant spectral profile?

LSD is lower is better. LSD does not change rapidly in early training because spectral shape
learning follows the denoising task - the model first learns to suppress noise broadly,
then refines spectral accuracy. Flat LSD in early epochs is expected and not a problem.

### 5.4 SI-SNR Metric

SI-SNR is kept as a monitoring metric (not primary save monitor) because it is the most direct
measurement of the training objective. The loss directly optimizes SI-SNR, so tracking it
validates that the loss is working. It also provides a reference for comparing to published
results - most speech enhancement papers report SI-SNRi as a primary metric.

### 5.5 Noise Removed Percentage

This metric computes what fraction of the original noise power has been eliminated. It is defined
as (original_noise_power - remaining_noise_power) / original_noise_power × 100. Negative values
mean the model is adding noise - expected in the first 5-10 epochs when weights are random.

This metric is the closest approximation to what an end user would call "how clean does it sound."
It is a power-domain measurement that directly reflects the denoising task performance.

Importantly, this metric is a proxy for CBAK (background noise quality) in the standard
VoiceBank-DEMAND evaluation suite. CBAK specifically measures how much background noise intrudes
into the enhanced signal. noise_removed_% is computed without an external library and provides
the same qualitative information.

### 5.6 SISNR Diagnostic Per-Sample

The SISNR diagnostic captures fixed validation samples to track per-sample performance across
epochs. This reveals whether the model is improving on all samples or degrading some while
improving others.

Deeply negative SISNR on a sample (-13 to -30 dB) indicates the model is making that specific
sample significantly worse than the input. This is a critical diagnostic for reverb failure:
reverb samples consistently show negative SISNR in early training, and the diagnostic tracks
when (if ever) they turn positive.

The alternating pattern between samples (one positive, one negative at early epochs) reflects
the different augmentation strategies applied to different validation files - some files have
reverb, some don't, and the model's performance gap between them reveals its reverb capability.

---

## 6. Dataset and Preprocessing Decisions

### 6.1 VoiceBank-DEMAND 28spk

VoiceBank-DEMAND is the standard benchmark for monaural speech enhancement. It contains 28
speakers (training) and 2 speakers (test) with real recordings from the VCTK corpus mixed with
noise from the DEMAND database at 4 SNR levels (0, 5, 10, 15 dB). Using the 28-speaker training
set maximizes speaker diversity, which is critical for generalization - a model trained on 2
speakers will learn speaker-specific noise patterns rather than general noise suppression.

The fixed train/validation split (valid_pct=0.15) gives 9837 training and 1735 validation pairs.
The validation set is used to monitor training progress. A fixed held-out test set (no augmentation,
not seen during training) is needed for final absolute performance evaluation.

### 6.2 Target Length 40000 (2.5 seconds)

2.5-second clips at 16kHz (40000 samples) were chosen to balance context length against VRAM usage.
The model needs enough context to capture sentence-level prosodic patterns (500ms-1s) and reverb
tails (200-600ms). 2.5 seconds provides substantial context while fitting in 4GB VRAM at
batch_size=2, channels=64, num_blocks=11, num_repeats=2.

The OOM limit was empirically found at target_length=48000 with this configuration, so 40000
is the safe maximum.

### 6.3 Why Not 44.1kHz for Speech

16kHz is the standard sample rate for speech enhancement research. Speech intelligibility is
primarily carried in frequencies up to 8kHz (the Nyquist for 16kHz). Higher sample rates add
computational cost without improving the speech enhancement task on current architectures. For
vocal music enhancement, a higher sample rate (22.05kHz minimum) would be needed to preserve
the harmonic content above 8kHz.

### 6.4 Validation Split – Fixed Seed Required

The validation set split must use a **constant seed** (e.g., `seed=42`), NOT `EPOCH_SEED.value`.
Using a varying seed changes the validation composition every epoch, introducing noise into
metric comparisons. A fixed split ensures epoch‑to‑epoch comparability:
- Validation metrics reflect genuine model improvement, not data‑split variation.
- Curriculum augmentation (which *does* use EPOCH_SEED per file) remains independent.

Implementation in `DataBlock`:
```python
splitter=RandomSplitter(valid_pct=valid_pct, seed=42)  # fixed, not EPOCH_SEED.value

---

## 7. Normalization and Inference Pipeline

### 7.1 Why Normalization Is External to the Model

The normalization step (measure input RMS, scale to -18 dBFS, process, restore original level)
adds approximately 2 multiplications and 1 RMS computation - unmeasurable latency compared to
the model inference. Keeping it external means the model's 216k parameters are entirely dedicated
to the enhancement task rather than learning a trivial amplitude mapping.

Training the model to learn normalization would require exposing it to a wide range of input
amplitudes. This would require amplitude augmentation in the dataset, which would add training
variance and slow convergence. The explicit normalization step eliminates this problem entirely.

### 7.2 Real-Time Processing Architecture

The causal architecture enables frame-by-frame streaming processing:

```
chunk -> normalize to -18 dBFS -> model -> restore original level -> output
```

The normalization uses a sliding window RMS over the current chunk, not the global signal RMS.
This handles the non-stationarity of real speech - a quiet passage followed by a loud burst
is normalized independently per chunk, preventing the quiet passage from being over-amplified.

For the model's causal convolutions, the CausalConv1d implementation maintains a context buffer
(pad_cache) that carries information from the previous chunk. This is what makes streaming
possible - the model sees the correct historical context at every timestep without needing to
reprocess previous chunks.

---

## 8. What This Combination Achieves

The decisions described above form a coherent system where each component supports the others.

The architecture (causal, deep, gated, stride-2) provides the representational capacity to learn
complex noise-speech boundaries across time while fitting in 4GB VRAM. The loss hierarchy
(SI-SNR primary, complex spectral secondary, temporal tertiary) provides gradient signals at
multiple levels of abstraction - global waveform quality, frequency-domain accuracy, temporal
structure, and artifact suppression. The curriculum ensures the model builds these skills in the
right order, with appropriate output scaling for each task's difficulty. The normalization
ensures the model operates in a consistent amplitude regime at all times.

The result is a model that can be applied directly to microphone audio in real time, producing
speech that is cleaner and more natural than the noisy input, while remaining small enough
(216k parameters) to run efficiently on consumer hardware including the AMD RX 560X via DirectML.

The realistic performance ceiling for this model and hardware is approximately PESQ 2.6-2.8,
SI-SNRi 12-14 dB, noise_removed 70-80%. Achieving PESQ above 3.0 would require either a larger
model, a non-causal architecture, or adversarial training - all requiring significantly more
compute than is available on this hardware.
