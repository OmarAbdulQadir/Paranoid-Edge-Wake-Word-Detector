# tinyML.md — Embedded / MCU Implementation

This document covers the firmware and hardware side of the Marvin project:
microphone integration, the audio capture pipeline, on-device feature
extraction, model deployment via X-CUBE-AI, memory budgeting, and the
real-time streaming inference loop. It's the counterpart to `Model.md`,
which covers the ML side — the model described there is designed against
the constraints described here, not the other way around.

## Hardware summary

| Component | Detail |
|---|---|
| MCU | STM32F401, Cortex-M4F, 84 MHz |
| Microphone | INMP441, digital MEMS, I2S output |
| Model deployment | X-CUBE-AI (generates C inference code from the quantized model) |
| Host communication | USART2 @ 115200 baud |

## Microphone: INMP441 over I2S

The INMP441 is a digital MEMS microphone with a native I2S output — no
analog front-end or external ADC needed, which is why it was chosen over
an analog electret + ADC approach: it maps directly onto the STM32F401's
I2S peripheral with no additional signal-conditioning hardware.

Key integration points:

- **Peripheral**: the F401's I2S functionality is implemented via the
  SPI2 or SPI3 peripheral in I2S mode (Philips/I2S standard protocol),
  avoiding any dependency on DFSDM (which the F401 doesn't have in a form
  usable for this) — this was confirmed during scoping as a hard
  compatibility requirement before committing to this mic
- **Clocking**: 16 kHz sample rate generated via PLLI2S, matching the
  sample rate the model was trained on (see `Model.md`) — this rate is
  fixed as a contract between the firmware and the training pipeline;
  changing one without the other silently breaks the whole system
- **Data format**: the INMP441 outputs 24-bit samples within a 32-bit I2S
  frame. The peripheral captures the full 32-bit word; firmware then
  extracts the 24-bit sample via shifting/masking rather than assuming
  the raw 32-bit word is directly usable audio
- **Channel selection**: the INMP441's L/R pin selects which half of the
  I2S frame it outputs data on (tied high or low for a single mono mic).
  Firmware must read the correct half of the frame — the other half
  carries no signal from this device
- **DMA double-buffering**: audio capture uses circular DMA into two
  alternating ("ping-pong") buffers, with half-transfer and
  transfer-complete interrupts marking when each buffer is ready for
  processing. This lets one buffer be consumed (feature extraction /
  streaming to host) while the other fills, with no gaps in capture — this
  same buffering pattern is reused later for the streaming inference
  window in Phase 3, so it's built once and treated as a shared
  foundation rather than rebuilt per phase

## Phase 2: capture validation (before any model involvement)

Before feature extraction or inference touch the audio, Phase 2 exists
specifically to validate that the *captured* audio is trustworthy:

- Stream raw PCM over USART2 to a PC, reconstruct it as a WAV file, and
  listen to it / inspect its waveform and spectrogram
- Check for clipping, DC offset, and noise floor issues introduced by the
  capture path itself (as distinct from problems in the model or the
  DSP port)
- Compare the spectral characteristics of MCU-captured audio against a
  Speech Commands sample of the same word — since the model was trained
  on dataset audio, a large distribution mismatch here predicts poor
  on-device accuracy *before* wiring up inference, rather than after
- Physical factors specific to the INMP441 (bottom-port MEMS mic) —
  enclosure design, orientation, distance from the sound source — are
  tuned at this stage, since they affect the raw signal the entire
  downstream pipeline depends on

## On-device feature extraction: CMSIS-DSP

The MFCC pipeline described in `Model.md` (40ms window, 20ms hop, 40 mel
bins, 10 coefficients) is re-implemented in C using CMSIS-DSP, targeting:

- `arm_rfft_fast_f32` (or a fixed-point variant, if cycle budget demands
  it) for the FFT stage
- A precomputed Mel filterbank matrix (computed once offline, stored as a
  constant — not recomputed on-device)
- DCT-II for the final coefficient extraction

**Validation approach**: the same WAV file is run through both the Python
reference implementation and this C implementation, and the numeric output
is compared directly, before ever trusting this pipeline against live mic
input. This isolates DSP-parity bugs from microphone or model issues,
which is why it's sequenced as its own explicit step rather than folded
into general "does it work" testing.

## Model deployment: X-CUBE-AI

The quantized (INT8) `.tflite` model from Phase 1 is imported into
X-CUBE-AI, which:

- Generates C code for the inference pass, callable from firmware
- Reports the **actual activation/tensor-arena RAM** requirement — this
  is the authoritative number for the 20KB activation budget (the
  `.tflite` file size alone only bounds the weight storage, not the RAM
  needed during inference), so this import step is treated as a gate
  before Phase 1 is considered complete
- Weight storage target: <30KB (flash)
- Activation RAM target: <20KB (RAM, during inference)

## Streaming inference loop (Phase 3)

Once capture (Phase 2) and the DSP port are both validated independently,
they're combined into a continuous pipeline:

1. DMA fills the ping-pong audio buffers continuously (as in Phase 2)
2. A sliding window over recent audio feeds the CMSIS-DSP MFCC extraction
3. The resulting feature frame is passed to the X-CUBE-AI-generated
   inference function
4. **Confidence smoothing** is applied across consecutive predictions
   (a moving average or majority vote over N frames) before declaring a
   detection — a single high-confidence frame is not sufficient, since
   isolated spurious activations are far more likely than sustained ones;
   this directly reuses the same decision-threshold concept characterized
   in `Model.md`'s evaluation section
5. On a confirmed detection, the result is sent over USART2 to the host

**Timing budget**: MFCC extraction + inference must comfortably fit within
the inference cadence (roughly the 20ms hop interval, though the effective
detection rate can be coarser depending on smoothing window size) on an
84 MHz Cortex-M4F. This is profiled explicitly (cycle counts for DSP stage
vs. inference stage) rather than assumed to "probably fit" — if it doesn't,
the options are reducing MFCC resolution, reducing model width, or
adjusting the smoothing window, and profiling data is what decides between
them rather than guessing.

## On-device evaluation

The same FAR/FRR methodology from `Model.md` is re-run here, but against
live spoken audio rather than dataset WAV files. A gap between the PC-side
numbers (Phase 1) and the on-device numbers (Phase 3) is expected and
informative — it's the signal used to tell apart:

- a **DSP-parity problem** (C MFCC output doesn't match Python reference)
- a **capture/hardware problem** (mic placement, noise floor, clipping —
  see Phase 2)
- a **distribution-shift problem** (real speech simply differs from the
  dataset's crowd-sourced recordings in ways augmentation didn't cover)

Because Phases 1 and 2 are validated independently first, a large gap at
this stage can be triaged against those earlier results rather than
debugged from scratch.

## Optional: audio output (Phase 4)

If pursued, response audio (tone/beep or short recorded clips) is driven
via PWM or DAC + timer-triggered DMA — architecturally a mirror of the
capture DMA pattern already built for the microphone, just in the output
direction, and deliberately scoped as a separate, optional subsystem
rather than a dependency of the core detection pipeline.
