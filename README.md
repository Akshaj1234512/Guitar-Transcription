# Music AI

**Setup Instructions:**

Set up a new Conda environment

```
conda env create -f environment.yml
conda activate guitar
```

Then, download the models as below.
```
cd ~/Music-AI/models
hf download shamakg/audio_to_midi_guitar --local-dir audio_to_midi
hf download shamakg/string-fret-guitar --local-dir string-fret
hf download shamakg/expressive-techniques-guitar --local-dir expressive-techniques-guitar
```

If you get an error relating to too many API calls being made, you can use:
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


### Project Description


### Key Resources


### Members
Akshaj, Andrea, Peter, Shamak, Samhita, Jiachen, Robbie
