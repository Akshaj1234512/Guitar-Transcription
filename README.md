# TART: Technique-Aware Audio-to-Tab Guitar Transcription

End-to-end guitar tablature transcription from raw audio. TART takes a guitar recording and produces a playable tablature with correct onset, pitch, string, fret, and expressive technique labels.

## Overview

Automatic Music Transcription has advanced significantly for the piano but remains limited for the guitar: existing systems often fail to detect expressive techniques (slides, bends, hammer-ons), frequently map notes to the wrong string and fret, and generalize poorly to recordings with real-world noise. TART addresses these gaps with a four-stage pipeline:

1. **Audio → MIDI.** A high-resolution CRNN transcribes guitar audio to MIDI notes, trained with a Stochastic Noise Augmentor to stay robust under varied recording conditions.
2. **Expressive technique classification.** A compact CNN-BiLSTM labels each note with techniques like hammer-on, bend, slide, or palm mute.
3. **String/fret assignment (AudioFret).** A T5-style encoder-decoder conditioned on per-note audio timbre + symbolic context resolves the pitch-redundancy problem, assigning each note to a specific string and fret.
4. **Tablature generation.** The combined output is rendered as standard notation (.musicxml) and timed MIDI.

The codebase contains everything needed to (a) deploy the pipeline on your own recordings and (b) reproduce every result from the paper.

## Quick start (inference)

### 1. Clone and install

```bash
git clone https://github.com/Akshaj1234512/Guitar-Transcription.git
cd Guitar-Transcription
conda env create -f environment.yml
conda activate new_venv
```

### 2. Download pretrained models

Stage 1 (audio → MIDI) and Stage 2 (expressive techniques):

```bash
mkdir -p models && cd models
hf download shamakg/audio_to_midi_guitar --local-dir audio_to_midi
hf download shamakg/expressive-techniques-guitar --local-dir expressive-techniques-guitar
```

If you hit a HuggingFace rate limit (anonymous users are capped at ~1000 requests/5min), either run `hf auth login` first or retry each download individually with `--max-workers 1`:

```bash
hf download shamakg/audio_to_midi_guitar --local-dir audio_to_midi --max-workers 1
hf download shamakg/expressive-techniques-guitar --local-dir expressive-techniques-guitar --max-workers 1
```

Stage 3 (AudioFret) — **[TODO: upload checkpoint to HuggingFace and replace this block]**

```bash
# Once uploaded:
# hf download <username>/audiofret --local-dir string-fret --max-workers 1

# For now, place the file manually:
mkdir -p string-fret
cp ../checkpoints/audiofret.pt string-fret/audiofret.pt
```

### 3. Transcribe a file

```bash
python predict.py --audio_path /path/to/guitar.wav
```

Outputs land in `results/` as `.xml`, `.musicxml`, `.jams`, and `.mid`.

### 4. Batch transcribe

```bash
python batch_process_audio.py /path/to/folder1 /path/to/folder2
```

## Repository layout

```
Music-AI/
├── predict.py                    # single-file inference entry point
├── batch_process_audio.py        # batch version
├── pipeline_utils/               # Stage 1 + technique + AudioFret runners + tab gen
├── tab_generation_utils/         # MIDI → MusicXML / JAMS rendering
├── t5_fretting_transformer/      # AudioFret model, tokenizer, and constrained decoder
├── models/                       # downloaded pretrained checkpoints (gitignored)
├── checkpoints/                  # locally saved training checkpoints (see below)
├── environment.yml
│
├── training/                     # scripts to reproduce every trained model
│   ├── train_fret_t5.py              # tiny T5 (Fretting-Transformer baseline)
│   ├── train_scaled.py               # scaled T5 backbone (d=256, 6 layers)
│   ├── train_audio_conditioned.py    # per-note audio CNN string classifier
│   ├── finetune_audio_conditioned.py # full AudioFret end-to-end finetune
│   ├── prepare_finetune_data.py      # builds finetune manifests from GAPS/GOAT/Guitar-TECHS
│   └── data_loaders/                 # dataset adapters (GAPS, GOAT, Guitar-TECHS, etc.)
│
└── evaluation/                   # scripts to reproduce every reported number
    ├── evaluate_guitarset.py         # note-level metrics on GuitarSet
    ├── evaluate_egdb.py              # note-level metrics on EGDB
    ├── evaluate_frame_level.py       # TabCNN-style frame-level metrics
    ├── calculate_midi_f1.py          # mir_eval-based MIDI F1
    ├── eval_string_classifier.py     # standalone CNN evaluation
    ├── batch_stage1_only.py          # Stage 1 audio→MIDI inference
    ├── batch_stage3_only.py          # symbolic T5 (Fretting-Transformer) inference
    ├── batch_fusion_inference.py     # AudioFret end-to-end inference
    ├── oracle_inference.py           # AudioFret fed GT MIDI (isolates Stage 3)
    ├── batch_stage3_oracle.py        # symbolic T5 fed GT MIDI
    └── batch_cnn_oracle.py           # CNN-alone fed GT MIDI
```

## Results

All numbers are **zero-shot** (no finetuning on GuitarSet or EGDB at any stage) under a single fixed threshold configuration.

### Stage 1: Audio → MIDI (note-level F1)

| Model | GS clean | EGDB clean | GS noisy | EGDB noisy | Avg |
|---|---|---|---|---|---|
| FretNet        | 69.1 | 40.9 | 37.3 | 23.6 | 42.7 |
| NoteEM         | 82.9 | 59.0 | 70.0 | 67.6 | 69.9 |
| Riley et al.   | **88.1** | 68.9 | 74.2 | 67.5 | 74.7 |
| **TART (ours)** | 87.4 | **79.0** | **82.2** | **76.8** | **81.4** |

### Stage 3: String/fret assignment (Oracle Tab F1 — isolates Stage 3)

| Model | GS cln | GS noi | EG cln | EG noi | Avg |
|---|---|---|---|---|---|
| TabCNN\*             | —    | —    | 30.4 | 25.4 | 27.9 |
| Fretting-Transformer | 60.4 | 60.4 | 66.3 | 66.3 | 63.3 |
| CNN alone            | 54.8 | 55.2 | 64.8 | 58.9 | 58.4 |
| Scaled backbone      | 61.3 | 61.3 | 72.7 | 72.7 | 67.0 |
| **AudioFret (ours)** | **69.2** | **69.5** | **74.8** | **73.7** | **71.8** |

\*TabCNN released weights were trained on GuitarSet (Chen et al., 2024), so we report EGDB only for a fair zero-shot comparison.

### End-to-end (Tab F1, full pipeline audio → tab)

| Setting | GS cln | GS noi | EG cln | EG noi | Avg |
|---|---|---|---|---|---|
| **TART end-to-end** | 56.0 | 51.2 | 55.2 | 53.9 | **54.1** |
| Oracle upper bound (perfect Stage 1) | 69.2 | 69.5 | 74.8 | 73.7 | **71.8** |
| Propagation cost (Δ) | 13.2 | 18.3 | 19.7 | 19.8 | **17.8** |


## Reproducing the paper

1. **Stage 1 training**: see Riley et al.'s noise-augmented high-resolution CRNN. Our variant is packaged in `pipeline_utils/midi_utils/` with the Stochastic Noise Augmentor.
2. **Tiny T5 / scaled T5 pretraining**: `python training/train_fret_t5.py` or `train_scaled.py` on DadaGP+SynthTab manifests.
3. **String classifier pretraining**: `python training/train_audio_conditioned.py --phase 1` on GAPS+GOAT+Guitar-TECHS.
4. **AudioFret end-to-end finetune**: `python training/finetune_audio_conditioned.py --pretrained-checkpoint checkpoints/scaled_t5.pt --string-classifier-checkpoint checkpoints/string_classifier.pt`.
5. **Evaluate every row in the Stage 3 table**: run the oracle/end-to-end scripts in `evaluation/` against GuitarSet and EGDB annotations.

## Datasets used

| Stage | Training data | Evaluation data (held out) |
|---|---|---|
| Stage 1 | GAPS, GOAT, Guitar-TECHS, Leduc | GuitarSet, EGDB |
| Stage 2 | AGPT, IDMT, Magcil, Guitar-TECHS, EG-IPT | IDMT, Guitar-TECHS holdouts |
| Stage 3 | DadaGP, SynthTab (pretraining); GAPS+GOAT+Guitar-TECHS (finetune) | GuitarSet, EGDB |

Neither GuitarSet nor EGDB is used for training at any stage.

## Contributors

Akshaj, Andrea, Peter, Shamak, Samhita, Jiachen, Robbie
