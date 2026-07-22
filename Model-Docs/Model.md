# Model.md — DS-CNN Keyword Spotting Model

This document covers the machine learning side of the Marvin project: the
dataset, the feature representation, the model architecture, training
strategy, quantization, and evaluation methodology. It assumes the
hardware constraints described in `tinyML.md` as fixed inputs — this is
a model designed *for* a 30KB weight / 20KB activation-RAM budget on an
84 MHz Cortex-M4F, not a model that happens to be shrunk down after the
fact.

## Dataset

**Google Speech Commands v2** — 35 spoken words, ~105,829 utterances,
16 kHz mono, 1-second clips, from ~2,600 speakers, released by Google/AIY.
Chosen because it's the standard benchmark dataset for this exact task —
the "Hello Edge" paper and most subsequent MCU-KWS research use it, which
makes results comparable and the dataset's quirks (crowd-sourced audio,
uneven word counts, official speaker-independent splits) well understood.

The dataset provides its own `validation_list.txt` / `testing_list.txt`,
which assign entire speakers to a single split. Using these (rather than
a random shuffle-split) matters: without it, the same speaker's voice can
leak across train/val/test, silently inflating reported accuracy.

## Class scheme

Three classes, not a binary "Marvin vs. everything":

- **`marvin`** — the target wake word
- **`unknown`** — a sampled subset of the other 34 words in the dataset
- **`silence`** — 1-second crops from the dataset's background noise clips

This is the standard scheme used in the reference DS-CNN literature and in
Google's own Speech Commands training example, for a specific reason:
without an explicit `silence` class, a model trained only on
speech-vs-Marvin tends to interpret ambient noise as "not enough signal to
be confident" rather than confidently rejecting it — which produces worse
real-world FAR than treating silence as its own learned category.

`unknown` and `silence` are each sized *relative to* the `marvin` count
(not the full 34-word pool), so the model isn't overwhelmed by negative
examples relative to positives — the exact ratios are a tuning knob, set
initially and adjusted based on the first training run's confusion matrix.

## Feature extraction: MFCC

Raw audio isn't fed to the network directly — it's converted to Mel
Frequency Cepstral Coefficients (MFCCs) first, which is standard for
speech/audio classification because it compresses a 16,000-sample-per-second
signal into a much smaller, perceptually-relevant representation that a
tiny CNN can actually learn from within the parameter budget.

Parameters (matching the "Hello Edge" DS-CNN reference configuration):

| Parameter | Value |
|---|---|
| Sample rate | 16 kHz |
| Clip duration | 1.0 s |
| Window length | 40 ms (640 samples) |
| Hop length | 20 ms (320 samples) |
| Mel filterbank bins | 40 |
| MFCC coefficients kept | 10 |
| Resulting feature shape | 49 frames × 10 coefficients (490 values) |

Pipeline per frame: Hamming window → FFT → power spectrum → Mel filterbank
→ log → DCT-II → top 10 coefficients.

This exact computation is implemented twice in the project: once here in
Python (for training and PC-side validation) and once in C using
CMSIS-DSP on the MCU (see `tinyML.md`). The two must produce numerically
consistent output on the same input audio — this is the single most common
failure mode in embedded audio ML ("works on PC, garbage on device"), so
the Python version is written in explicit, auditable math (not a black-box
library call) specifically to make that cross-check tractable.

## Model architecture: DS-CNN

Depthwise-Separable CNN, following the "Hello Edge" topology:

```
Input: (49, 10, 1) MFCC feature map
  → Conv2D (10×4 kernel, stride 2×2) + BatchNorm + ReLU
  → 4× [ Depthwise Conv2D (3×3) + BN + ReLU
         → Pointwise Conv2D (1×1) + BN + ReLU ]
  → Global Average Pooling
  → Dense(3) + Softmax
```

**Why depthwise-separable over a regular CNN**: a standard convolution
mixes spatial and channel information in one operation, which is
parameter-expensive. Depthwise-separable convolution splits this into a
per-channel spatial filter (depthwise) followed by a 1×1 channel-mixing
filter (pointwise) — dramatically fewer parameters for similar
representational capacity. This is precisely the tradeoff that makes
DS-CNN outperform plain CNNs on accuracy-per-KB in the reference
benchmarks, which is the deciding factor under a hard 30KB ceiling.

**Sizing**: the network width (`num_filters`) is the primary knob for
trading capacity against the size budget. At 64 filters, the model lands
at roughly 22–23K trainable parameters (~22KB at INT8) — under the 30KB
ceiling with margin, and in the same range as the reference DS-CNN-S
configuration (~24K parameters) from the source paper. Narrower/wider
variants are evaluated if the initial accuracy/FAR/FRR numbers call for
it.

## Training

- **Loss**: sparse categorical cross-entropy over the 3 classes
- **Augmentation**: background-noise mixing (at low SNR, so speech stays
  dominant) and small random time-shifts (±100ms) — chosen specifically
  because they target the two things most different between the clean
  dataset audio and a real INMP441 capture: added noise floor and
  imprecise word-boundary timing
- **Monitoring**: FAR and FRR are computed on the validation set every
  epoch, not just accuracy/loss — a model can show improving accuracy
  while FAR quietly gets worse, and that would be invisible without
  tracking it directly
- **Regularization**: dropout before the final dense layer; early stopping
  and LR reduction on validation accuracy plateau

## Quantization

Post-training full-integer (INT8) quantization via the TFLite converter,
using a representative dataset (real samples drawn from the training set)
for activation range calibration — required for reasonable INT8 accuracy;
skipping calibration and using dynamic-range quantization instead tends to
degrade accuracy more than the budget savings are worth here.

If INT8 accuracy loss versus the float baseline turns out to be
unacceptable, the fallback is quantization-aware training (QAT) rather
than shipping a degraded model — this is a decision point to evaluate
with real numbers, not something to pre-decide.

The quantized model's file size gives a lower-bound check against the
30KB weight budget. The activation/tensor-arena RAM (the 20KB budget) is
*not* visible from the `.tflite` file size — that number only becomes
authoritative once the model is imported into X-CUBE-AI, which is why
that import is treated as part of Phase 1's "done" criteria, not deferred
to Phase 3.

## Evaluation methodology

- **FAR (False Acceptance Rate)**: fraction of non-Marvin audio (`unknown`
  + `silence`) incorrectly triggering a Marvin detection
- **FRR (False Rejection Rate)**: fraction of actual Marvin utterances
  missed
- These are computed against a tunable **decision threshold** on the
  Marvin class probability, not just `argmax` — the threshold is the same
  knob later used for confidence smoothing during streaming inference on
  the MCU (see `tinyML.md`), so it's characterized here first, on clean
  PC-side evaluation, before being re-tuned against live audio in Phase 3
- Reported first on the PC-side quantized model (Phase 1), then re-measured
  on-device against live spoken audio (Phase 3) — the gap between these two
  numbers is itself a useful signal, separating "model problem" from
  "capture/DSP-parity problem"
