# Music AI

Setup Instructions:

It is recommended to use a virtual environment. 

```
cd ~/Music-AI/
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install all dependencies from the requirements file
pip install --upgrade pip
pip install -r requirements.txt
```

Then, download the models as below.
```
cd ~/Music-AI/models
hf download shamakg/audio_to_midi_guitar --local-dir audio_to_midi
hf download shamakg/string-fret-guitar --local-dir string-fret
hf download shamakg/expressive-techniques-guitar --local-dir expressive-techniques-guitar
```


### Project Description

A next-generation music platform focused on guitar, combining AI-powered tools for transcription, learning, composition, and music generation. The platform aims to deliver:
- Transcription Services: Accurate audio-visual transcription of guitar performances into detailed tablature, capturing expressive techniques and finger positions.
- AI Tutor: Personalized, real-time feedback to help users improve guitar technique, adaptively tailored to individual progress.
- Music Generation: Intelligent algorithms to create original musical content and backing tracks based on user input or style preferences.
- Music Composition Tools: An intuitive environment for composing, arranging, and editing music, tightly integrated with transcription and generation features.

### Key Resources
Seminar Slideshow Presentation: https://docs.google.com/presentation/d/1dBwig8hyrlOXuLgaLisY6MGfUPYyA-XAe3ScQX5L9JQ/edit?slide=id.g36efb0cc50b_0_118#slide=id.g36efb0cc50b_0_118
Music AI Team Document: https://docs.google.com/document/d/11EGxikx48PIuqUgDy9gn_XR0xpirLDEiaLma8gGgU7Q/edit?tab=t.eons863jh8b8#heading=h.x4h19xf41lhh

### Members
Akshaj, Jiachen, Robbie, Anagha, Sid, Zane, Andrea, Neel
