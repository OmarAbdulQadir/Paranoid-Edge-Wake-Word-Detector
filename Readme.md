# Marvin — TinyML Keyword Spotting on STM32F401

## Overview

This project implements a single-keyword wake-word detector ("Marvin")
running entirely on an STM32F401 microcontroller (Cortex-M4F, 84 MHz), with
audio captured live from an INMP441 I2S MEMS microphone. It is the second
in a series of independent TinyML portfolio projects targeting this MCU —
following the Morse code decoder (1D-CNN, INT8, push-button input) — and is
not a continuation or dependency of that project; it stands on its own.

The goal is to demonstrate a complete, realistic embedded ML pipeline: from
dataset and model design, through quantization and size-budgeting for a
memory-constrained MCU, to real-time streaming inference against live
microphone audio, with measured, honest performance numbers rather than
just a demo that "seems to work."

## Research basis

The model architecture and feature pipeline follow the **DS-CNN
(Depthwise-Separable CNN) approach for keyword spotting**, as established
in Zhang, Suda, Lai, and Chandra, *"Hello Edge: Keyword Spotting on
Microcontrollers"* (2018). That paper benchmarks several neural network
topologies (DNN, CNN, RNN, CRNN, DS-CNN) specifically for MCU-class
resource budgets, and identifies DS-CNN as the best accuracy-per-parameter
tradeoff for this class of hardware — which is why it was chosen here over
a plain CNN. Where this project's design choices follow that paper
directly (MFCC framing, general topology shape), it's called out explicitly
in `Model.md`; where choices diverge (exact filter widths, class scheme,
confidence smoothing), the reasoning is explained rather than left
implicit.

## Problem framing

- **Task**: binary wake-word detection — did the user say "Marvin" or not
- **Classes**: `silence` / `unknown` / `marvin` (the standard 3-class KWS
  scheme — `unknown` and `silence` are both explicitly modeled negative
  classes, not just "everything else")
- **Dataset**: Google Speech Commands v2 (35 words, ~105K utterances,
  16 kHz mono WAV, 1-second clips)
- **Primary metrics**: False Acceptance Rate (FAR) and False Rejection
  Rate (FRR), not raw accuracy — for a wake-word product, *how* it fails
  matters as much as *how often*

## Hardware

| Component | Role |
|---|---|
| STM32F401 (Cortex-M4F, 84 MHz) | Inference + audio pipeline |
| INMP441 (I2S MEMS microphone) | Audio capture |
| X-CUBE-AI | Model deployment (Keras/TFLite → C inference code) |
| USART2 @ 115200 baud | Serial output / PC communication |

## Project structure

This documentation is split into three files:

1. **`Readme.md`** *(this file)* — project overview, motivation, research
   basis, hardware, and phase plan
2. **`Model.md`** — the ML side in detail: dataset handling, feature
   extraction, DS-CNN architecture, training, quantization, and evaluation
3. **`tinyML.md`** — the embedded/firmware side in detail: I2S microphone
   integration, DMA buffering strategy, CMSIS-DSP feature extraction port,
   X-CUBE-AI integration, memory budgeting, and real-time streaming
   inference on the MCU

## Development phases

The project is built in four phases, each independently validated before
moving to the next — deliberately avoiding the trap of debugging the model,
the microphone, and the firmware integration all at once.

| Phase | Description | Validates |
|---|---|---|
| **1. Model + PC inference** | Train and quantize the DS-CNN on Speech Commands v2, test inference on WAV files entirely on PC | The model itself: does it hit the accuracy/FAR/FRR/size targets, independent of any hardware? |
| **2. Microphone bring-up** | Capture audio via I2S/DMA on the MCU, stream it to a PC, listen to and analyze it | Is the MCU's captured audio actually clean enough to run inference on? |
| **3. Full on-device pipeline** | Port MFCC extraction to CMSIS-DSP, run live streaming inference on-device, output predictions over serial | The complete closed-loop system, with on-device FAR/FRR measured against live speech |
| **4. (Optional) Audio response** | Drive a speaker to respond audibly to a detected keyword, instead of only reporting over serial | End-to-end product feel, deferred until 1–3 are solid |

## Status

Phase 1 is in progress. See `Model.md` and `tinyML.md` for details on each
side of the pipeline as they're built out.
