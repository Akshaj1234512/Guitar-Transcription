# TART: Technique-Aware Audio-to-Tab Guitar Transcription

### Overview/Abstract
Automatic Music Transcription (AMT) has advanced significantly for the piano, but transcription for the guitar remains limited due to several key challenges. Current systems fail to detect expressive techniques (e.g., slides, bends, percussive hits) and often map notes to the incorrect string and fret combination in the generated tablature. Furthermore, prior models are typically trained on professionally recorded, isolated datasets, limiting their generalizability to varied acoustic environments with background noise, such as home recordings made on standard smartphones. To overcome these limitations, we propose TART, a four-stage end-to-end pipeline that produces detailed guitar tablature directly from guitar audio. Our system consists of (1) a CRNN-based audio-to-MIDI transcription model; (2) a CNN-BiLSTM for expressive technique classification; (3) a Transformer-based string and fret assignment model; and (4) an automated tablature generator, all consolidated into a pipeline that can output tablature from a given audio sample. To the best of our knowledge, this framework is the first to generate detailed tablature sheet music with accurate fingerings and expressive technique labels from guitar audio.

### Setup Instructions

Set up a new Conda environment:

```
conda env create -f environment.yml
conda activate new_venv
```

Download the pretrained models:
```
mkdir -p ~/Music-AI/models
cd ~/Music-AI/models
hf download shamakg/audio_to_midi_guitar --local-dir audio_to_midi
hf download shamakg/string-fret-guitar --local-dir string-fret
hf download shamakg/expressive-techniques-guitar --local-dir expressive-techniques-guitar
```

If you encounter API rate limit issues, use:
```
hf download shamakg/audio_to_midi_guitar \
  --local-dir audio_to_midi \
  --max-workers 1

hf download shamakg/string-fret-guitar \
  --local-dir string-fret \
  --max-workers 1

hf download shamakg/expressive-techniques-guitar \
  --local-dir expressive-techniques-guitar \
  --max-workers 1
```

**Usage:**

To generate .MIDI and .XML files use the command:
```
cd ~/Music-AI/
python predict.py --audio_path [PATH_TO_AUDIO]
```

For example:
```
cd ~/Music-AI/
python predict.py --audio_path /data/user/dataset/audio.wav
```

### Results

**Audio to MIDI (F1 Score)**

| Model           | GuitarSet | EGDB   | Noisy GuitarSet | Noisy EGDB |
|----------------|----------|--------|------------------|------------|
| FretNet        | 69.10%   | 40.90% | 37.30%           | 23.60%     |
| NoteEM         | 82.90%   | 59.00% | 70.00%           | 67.60%     |
| Riley et al.   | **88.10%** | 68.90% | 74.20%           | 67.50%     |
| TART + No Aug  | 87.70%   | 78.50% | 80.60%           | 76.10%     |
| TART + Aug     | 87.60%   | **78.50%** | **81.30%**       | **76.40%** |

**Technique Classification Results (F1 Score)**

We construct a unified dataset by aggregating multiple sources (AGPT, IDMT, Magcil, Guitar-TECHS, EG-IPT) and mapping their labels into a shared taxonomy.

| Model           | Params | Accuracy | Macro F1 |
|----------------|--------|----------|----------|
| Stefani et al. | 2.08M  | 86.7%    | 71.6%    |
| Fiorini et al. | 3.70M  | 64.3%    | 62.7%    |
| TART (Ours)    | **160K** | **97.4%** | **95.9%** |

**String–Fret Assignment Results (F1 Score)**

| Dataset                     | Tab Accuracy | Difficulty |
|----------------------------|--------------|------------|
| DadaGP                     | 81.24%       | 2.286      |
| SynthTab                   | **86.13%**   | 2.342      |
| GuitarSet (DadaGP base)    | 70.32%       | 3.908      |
| GuitarSet (SynthTab base)  | 72.79%       | 3.993      |

### Contributors
Akshaj, Andrea, Peter, Shamak, Samhita, Subhash, Jiachen, Robbie
