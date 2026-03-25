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

| Header 1 | Header 2 | Header 3 |
|---|---|---|
| Row 1, Col 1 | Row 1, Col 2 | Row 1, Col 3 |
| Row 2, Col 1 | Row 2, Col 2 | Row 2, Col 3 |


### Members
Akshaj, Andrea, Peter, Shamak, Samhita, Jiachen, Robbie
