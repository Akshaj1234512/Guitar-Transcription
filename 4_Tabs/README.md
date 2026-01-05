# JAMS to MusicXML Converter for Guitar Tablature

This Jupyter notebook provides a complete pipeline for converting guitar tablature data from the JAMS (JSON Annotated Music Specification) format to MusicXML, with support for guitar-specific playing techniques.

## Overview

The conversion pipeline follows these main steps:

1. **Load JAMS data** (from GuitarSet format or MIDI)
2. **Create tab_note annotation** with string/fret positions
3. **Add playing techniques** (bends, hammer-ons, pull-offs, etc.)
4. **Convert to MusicXML** with proper notation
5. **Post-process XML** for Guitar Pro compatibility

---

## Dependencies

```python
import jams
import numpy as np
import pretty_midi
from music21 import stream, note, tempo, meter, clef, articulations, expressions, spanner, interval, instrument
from lxml import etree
```

---

## Constants

### `STANDARD_TUNING`

A dictionary defining standard guitar tuning as MIDI pitch numbers for each string.

```python
STANDARD_TUNING = {
    6: 40,  # E2 (low E string)
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64   # E4 (high e string)
}
```

**Usage:** Referenced throughout the codebase to calculate fret positions from MIDI pitches.

---

## Functions

### 1. `midi_pitch_to_guitar_positions()`

```python
def midi_pitch_to_guitar_positions(midi_pitch, tuning=STANDARD_TUNING, max_fret=12)
```

**Purpose:** Finds all possible (string, fret) combinations where a given MIDI pitch can be played on the guitar.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `midi_pitch` | int | MIDI note number (e.g., 60 = middle C) |
| `tuning` | dict | Dictionary mapping string number to open string MIDI pitch |
| `max_fret` | int | Maximum fret to consider (default: 12) |

**Returns:** List of `(string, fret)` tuples, sorted by playability preference.

**How it works:**
1. Iterates through all 6 strings
2. Calculates the fret needed on each string: `fret = midi_pitch - open_string_pitch`
3. Filters positions where fret is within valid range (0 to max_fret)
4. Sorts results using a scoring function that prefers:
   - Middle strings (3-4) over outer strings
   - Lower frets over higher frets

**Example:**
```python
>>> midi_pitch_to_guitar_positions(60)  # Middle C
[(3, 5), (4, 10), (2, 1)]  # G string fret 5, D string fret 10, B string fret 1
```

---

### 2. `choose_best_position()`

```python
def choose_best_position(midi_pitch, previous_position=None, tuning=STANDARD_TUNING)
```

**Purpose:** Selects the optimal string/fret position for a note, considering the previous note's position to minimize hand movement.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `midi_pitch` | int | MIDI note number |
| `previous_position` | tuple | Previous `(string, fret)` position, or None |
| `tuning` | dict | Guitar tuning dictionary |

**Returns:** A single `(string, fret)` tuple representing the best position.

**How it works:**
1. Gets all possible positions via `midi_pitch_to_guitar_positions()`
2. If no previous position exists, returns the most natural position (first in sorted list)
3. If previous position exists, calculates a distance score for each option:
   - `score = fret_distance × 2 + string_distance`
   - Fret movement is weighted more heavily (harder than string changes)
4. Returns the position with the minimum distance score
5. Handles out-of-range pitches by finding the closest approximation

---

### 3. `midi_to_jams_with_tablature()`

```python
def midi_to_jams_with_tablature(midi_path, tuning=STANDARD_TUNING)
```

**Purpose:** Converts a MIDI file to JAMS format with intelligent tablature mapping, extracting tempo and time signature from the MIDI file.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `midi_path` | str | Path to the MIDI file |
| `tuning` | dict | Guitar tuning dictionary |

**Returns:** JAMS object containing:
- `note_midi` annotation (raw MIDI note data)
- `tab_note` annotation (with string, fret, and technique fields)
- Metadata: tempo_bpm, time_signature, duration

**How it works:**
1. Loads MIDI file using `pretty_midi`
2. Extracts tempo from MIDI tempo changes (defaults to 120 BPM)
3. Extracts time signature (defaults to 4/4)
4. Creates JAMS object with file metadata
5. Adds `note_midi` annotation with all notes from the first instrument
6. Creates `tab_note` annotation by:
   - Processing notes sequentially
   - Using `choose_best_position()` to map each pitch to string/fret
   - Tracking previous position for intelligent fingering

**Output structure (tab_note value):**
```python
{
    "pitch": 60,           # MIDI pitch
    "string": 3,           # Guitar string (1-6)
    "fret": 5,            # Fret number (0-24)
    "techniques": []      # List of techniques
}
```

---

### 4. `guitarset_jams_to_tab_note()`

```python
def guitarset_jams_to_tab_note(jam)
```

**Purpose:** Converts GuitarSet's native JAMS format (which stores notes separately per string) into a unified `tab_note` annotation.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `jam` | JAMS | JAMS object in GuitarSet format |

**Returns:** Modified JAMS object with added `tab_note` annotation.

**How it works:**
1. Finds all `note_midi` annotations (GuitarSet has 6, one per string)
2. For each string's annotation:
   - Calculates string number from index (index 0 = string 6 = low E)
   - Computes fret position: `fret = round(midi_pitch) - open_string_pitch`
   - Clamps fret to valid range (0-24)
3. Combines all notes into a single list, sorted by time
4. Creates unified `tab_note` annotation
5. Extracts tempo from `tempo` annotations if present
6. Extracts time signature from `beat_position` annotations if present

**Note:** This function is specifically designed for the GuitarSet dataset format where each guitar string has its own annotation track.

---

### 5. `add_random_techniques_to_existing_jam()`

```python
def add_random_techniques_to_existing_jam(jam, technique_probability=0.5)
```

**Purpose:** Adds random guitar playing techniques to an existing `tab_note` annotation. Useful for testing or generating synthetic training data.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `jam` | JAMS | JAMS object with existing `tab_note` annotation |
| `technique_probability` | float | Probability (0-1) of adding a technique to each note |

**Returns:** Modified JAMS object with techniques added.

**Supported techniques:**
- `"vibrato"` - Vibrato effect
- `"hammer-on"` - Hammer-on legato technique
- `"pull-off"` - Pull-off legato technique
- `"bend"` - String bend
- `"harmonic"` - Natural or artificial harmonic

**How it works:**
1. Finds the existing `tab_note` annotation
2. Iterates through each note
3. For each note, randomly decides whether to add a technique (based on probability)
4. If adding, randomly selects one technique from the available options
5. Modifies the note's `techniques` list in place

---

### 6. `jams_to_musicxml_real()`

```python
def jams_to_musicxml_real(jam, output_xml='output.xml', tempo_bpm=120, title=None, composer=None)
```

**Purpose:** The main conversion function that transforms a JAMS object with tablature data into a properly formatted MusicXML file.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `jam` | JAMS | JAMS object with `tab_note` annotation |
| `output_xml` | str | Output file path |
| `tempo_bpm` | int | Tempo in beats per minute |
| `title` | str | Title for the score (optional) |
| `composer` | str | Composer name (optional) |

**Returns:** Path to the created MusicXML file.

**How it works:**

1. **Setup:**
   - Finds the `tab_note` annotation
   - Creates a music21 Score with metadata
   - Sets up guitar instrument, tempo, time signature, and tab clef

2. **Note quantization:**
   - Quantizes note times to nearest eighth note
   - Quantizes durations (minimum: eighth note)
   - Calculates total measures needed

3. **Measure processing:**
   - Groups notes by measure
   - Groups simultaneous notes as chords
   - Inserts rests for gaps between notes
   - Handles measure boundaries correctly

4. **Technique processing:**
   - **Hammer-on/Pull-off:** Records note indices for post-processing
   - **Bend:** Creates `FretBend` articulation with bend amount
   - **Slide:** Adds text expression "/"
   - **Vibrato:** Adds fingering marker for post-processing
   - **Harmonic:** Adds `Harmonic` articulation

5. **Output:**
   - Writes MusicXML using music21
   - Calls `post_process_musicxml()` with technique map for final adjustments

**Chord handling:** Notes at the same time position are automatically grouped into chords.

---

### 7. `post_process_musicxml()`

```python
def post_process_musicxml(xml_path, technique_map=None, tuning=None)
```

**Purpose:** Post-processes the MusicXML file to add proper Guitar Pro-compatible elements for techniques that can't be fully represented by music21.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `xml_path` | str | Path to the MusicXML file |
| `technique_map` | dict | Maps note indices to technique info |
| `tuning` | list | MIDI pitches for strings (default: standard tuning) |

**technique_map format:**
```python
{
    note_index: {
        'technique': 'hammer-on' | 'pull-off',
        'is_target': True  # Note is the destination of the technique
    }
}
```

**How it works:**

1. **Parses XML** using lxml for direct manipulation

2. **Helper functions:**
   - `pitch_to_midi()`: Converts MusicXML pitch element to MIDI number
   - `midi_to_string_fret()`: Maps MIDI to guitar position

3. **Collects all notes** (excluding chord notes and rests)

4. **Processes hammer-ons/pull-offs:**
   - Adds `<hammer-on>` or `<pull-off>` elements with start/stop types
   - Adds `<slur>` elements to connect the notes visually
   - Source note gets `type="start"`, target note gets `type="stop"`

5. **Processes other techniques:**
   - **Harmonics:** Adds direction text "Harm." above the note
   - **Vibrato:** Adds Guitar Pro processing instruction

6. **Sets instrument metadata:**
   - Part name: "Classical Guitar"
   - Instrument sound: "pluck.guitar.nylon-string"

7. **Writes modified XML** back to file

---

### 8. `optimize_tablature_positions()`

```python
def optimize_tablature_positions(jam, max_stretch=4)
```

**Purpose:** Re-optimizes string/fret positions for better playability by keeping notes within reasonable hand positions.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `jam` | JAMS | JAMS object with `tab_note` annotation |
| `max_stretch` | int | Maximum fret stretch (default: 4 frets) |

**Returns:** Modified JAMS object with optimized positions.

**How it works:**
1. Tracks current hand position (center fret)
2. For each note:
   - Gets all possible positions for the pitch
   - Filters to positions within `max_stretch` of current position
   - If valid positions exist, chooses the closest one
   - If no positions within reach, shifts hand position to the new fret
3. Updates the note's string and fret values in place

**Use case:** Call this after initial tablature generation to improve fingering ergonomics.

---

## Complete Usage Example

```python
# Step 1: Load GuitarSet JAMS file and create tab_note annotation
jam = jams.load('00_BN1-129-Eb_comp.jams')
jam = guitarset_jams_to_tab_note(jam)

# Step 2: (Optional) Add techniques to notes
jam = add_random_techniques_to_existing_jam(jam, technique_probability=0.8)

# Step 3: (Optional) Optimize fingering positions
jam = optimize_tablature_positions(jam, max_stretch=4)

# Step 4: Convert to MusicXML
bpm = getattr(jam.sandbox, 'tempo_bpm', 120)
jams_to_musicxml_real(
    jam, 
    'output.musicxml', 
    tempo_bpm=bpm, 
    title='My Piece', 
    composer='Artist Name'
)
```

---

## Supported Techniques

| Technique | MusicXML Representation | Notes |
|-----------|------------------------|-------|
| `hammer-on` | `<hammer-on>` + `<slur>` | Legato articulation ascending |
| `pull-off` | `<pull-off>` + `<slur>` | Legato articulation descending |
| `bend` | `<bend>` with alter value | Currently hardcoded to half-step |
| `vibrato` | Guitar Pro processing instruction | Rendered as "Slight" vibrato |
| `harmonic` | `<harmonic>` + "Harm." text | Natural harmonic indication |
| `slide` | Text expression "/" | Basic slide notation |

---

## Output Format

The generated MusicXML files include:
- Tab clef notation
- Guitar instrument definition (Classical Guitar)
- Tempo and time signature markings
- String and fret numbers for each note
- Technical notations for guitar techniques
- Proper measure structure with rests

Files are compatible with:
- MuseScore
- Guitar Pro
- Finale
- Sibelius
- Other MusicXML-compatible notation software

---

## Notes and Limitations

1. **Quantization:** Notes are quantized to eighth-note precision
2. **Bend amount:** Currently hardcoded to half-step bends
3. **Chord techniques:** Techniques on chord notes may not be fully supported
4. **Tuning:** Standard tuning is assumed; alternate tunings require modification
5. **Fret range:** Default maximum fret is 12 (can be adjusted)
