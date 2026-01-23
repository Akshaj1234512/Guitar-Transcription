import jams
from lxml import etree
from collections import defaultdict

# ----------------------------
# Settings
# ----------------------------
TIME_SIG_NUM = 8
TIME_SIG_DEN = 4
DIVISIONS = 480               # per quarter note

GRID_DENOM = 16               # snap to 1/16 notes
GRID_TICKS = int(DIVISIONS * (4 / GRID_DENOM))  # ticks per 1/16

PAIR_WINDOW_S = 0.6

# IMPORTANT: MuseScore TAB tends to treat staff-tuning line=1 as LOWEST string (E2).
# So we provide tunings LOW->HIGH for line 1..6:
STAFF_TUNING_LINE_1_TO_6_LOW_TO_HIGH = [
    ("E", 2), ("A", 2), ("D", 3), ("G", 3), ("B", 3), ("E", 4)
]

# If your JAMS uses 1=LOW E ... 6=HIGH E (common), and you want MusicXML "string"
# values to follow the SAME convention for MuseScore (line1=low), then DO NOT flip.
# Set this to True only if your JAMS is 1=HIGH e ... 6=LOW E.
JAMS_IS_1_HIGH = False


# ----------------------------
# Helpers
# ----------------------------
def midi_pitch_to_step_alter_oct(midi_pitch: int):
    steps = ['C','C','D','D','E','F','F','G','G','A','A','B']
    alters = [0,1,0,1,0,0,1,0,1,0,1,0]
    pc = midi_pitch % 12
    return steps[pc], alters[pc], (midi_pitch // 12) - 1

def normalize_techs(v):
    return [str(t).lower().strip() for t in (v.get("techniques", []) or []) if str(t).strip()]

def seconds_to_quarters(sec, tempo_bpm):
    return sec * (tempo_bpm / 60.0)

def quantize_to_grid_ticks(sec, tempo_bpm):
    """Quantize absolute time to nearest 1/16 grid."""
    qn = seconds_to_quarters(sec, tempo_bpm)
    ticks = int(round(qn * DIVISIONS))
    return int(round(ticks / GRID_TICKS)) * GRID_TICKS

def quantize_dur_to_grid_ticks(sec, tempo_bpm):
    """Quantize duration to nearest 1/16 grid, minimum one grid."""
    qn = seconds_to_quarters(sec, tempo_bpm)
    ticks = int(round(qn * DIVISIONS))
    qt = int(round(ticks / GRID_TICKS)) * GRID_TICKS
    return max(GRID_TICKS, qt)

def quantize_to_grid_ticks_floor(sec, tempo_bpm):
    """Snap absolute time DOWN to the 1/16 grid (more chord-friendly than round)."""
    qn = seconds_to_quarters(sec, tempo_bpm)
    ticks = int(round(qn * DIVISIONS))
    return (ticks // GRID_TICKS) * GRID_TICKS


def duration_to_type(dur_ticks):
    # rough visual type only; MuseScore is fine if this is imperfect
    if dur_ticks >= DIVISIONS * 4: return "whole"
    if dur_ticks >= DIVISIONS * 2: return "half"
    if dur_ticks >= DIVISIONS:     return "quarter"
    if dur_ticks >= DIVISIONS//2:  return "eighth"
    if dur_ticks >= DIVISIONS//4:  return "16th"
    if dur_ticks >= DIVISIONS//8:  return "32nd"
    return "64th"

def ensure_notations(note_elem):
    n = note_elem.find("notations")
    if n is None:
        n = etree.SubElement(note_elem, "notations")
    return n

def ensure_technical(notations_elem):
    t = notations_elem.find("technical")
    if t is None:
        t = etree.SubElement(notations_elem, "technical")
    return t

def add_staff_details(attributes_elem):
    clef = etree.SubElement(attributes_elem, "clef")
    etree.SubElement(clef, "sign").text = "TAB"
    etree.SubElement(clef, "line").text = "5"

    sd = etree.SubElement(attributes_elem, "staff-details")
    etree.SubElement(sd, "staff-lines").text = "6"

    # line="1" .. line="6"
    for line, (step, octave) in enumerate(STAFF_TUNING_LINE_1_TO_6_LOW_TO_HIGH, start=1):
        st = etree.SubElement(sd, "staff-tuning", line=str(line))
        etree.SubElement(st, "tuning-step").text = step
        etree.SubElement(st, "tuning-octave").text = str(octave)

def jams_string_to_xml_string(jams_string: int) -> int:
    # MuseScore expects "string" number to match staff-tuning line.
    # We set line 1 = low E ... line 6 = high e.
    # If JAMS is also 1=low..6=high -> keep. If reversed -> flip.
    s = int(jams_string)
    return (7 - s) if JAMS_IS_1_HIGH else s


# ----------------------------
# Pair spanners: LEGATO + SLIDE
# ----------------------------
def build_pairs(tab_data, max_dt=0.6, max_fret_jump=7):
    by_string = defaultdict(list)
    for i, obs in enumerate(tab_data):
        by_string[int(obs.value["string"])].append(i)

    pairs = {}
    for s, idxs in by_string.items():
        for a, b in zip(idxs, idxs[1:]):
            oa, ob = tab_data[a], tab_data[b]
            dt = float(ob.time) - float(oa.time)
            if dt > max_dt:
                continue

            techs = set(normalize_techs(oa.value))
            if not techs:
                continue

            fa = oa.value.get("fret", None)
            fb = ob.value.get("fret", None)
            if fa is not None and fb is not None:
                if abs(int(fb) - int(fa)) > max_fret_jump:
                    continue

            if "hammer_on_pull_off" in techs or "hopo" in techs or "hammer" in techs or "pull" in techs:
                pairs.setdefault(a, {})["legato"] = b

            if "slide" in techs:
                pairs.setdefault(a, {})["slide"] = b

    return pairs


# ----------------------------
# Main writer
# ----------------------------
def jams_to_musicxml_tab_musescore_clean(
    jams_path,
    output_xml="final_tab.musicxml",
    tempo_bpm=120,
    pair_window_s=PAIR_WINDOW_S,
    add_slide_slur=False,
):
    jam = jams.load(jams_path, validate=False)
    tab_ann = next((a for a in jam.annotations if a.namespace == "tab_note"), None)
    if tab_ann is None:
        raise ValueError("No tab_note annotation found.")

    tab_data = sorted(tab_ann.data, key=lambda o: (float(o.time), int(o.value["string"])))
    pairs = build_pairs(tab_data, max_dt=pair_window_s)

    # Build events in tick-time (SNAP to 1/16 grid)
    events = []
    for j_idx, obs in enumerate(tab_data):
        v = obs.value
        pitch = int(v["pitch"])
        xml_string = jams_string_to_xml_string(int(v["string"]))

        # KEEP fret from JAMS if present (recommended)
        if ("fret" in v) and (v["fret"] is not None):
            fret = int(v["fret"])
        else:
            fret = 0  # fallback

        start = quantize_to_grid_ticks_floor(float(obs.time), tempo_bpm)
        dur   = quantize_dur_to_grid_ticks(float(obs.duration), tempo_bpm)


        techs = set(normalize_techs(v))
        # constraints
        techs.discard("palm_muting"); techs.discard("pm"); techs.discard("bend")

        events.append(dict(
            j_idx=j_idx,
            start=start,
            dur=dur,
            pitch=pitch,
            xml_string=xml_string,
            fret=fret,
            techs=techs
        ))

    # REMOVE any notes that collide with previous notes on same (start, string)
    # Keep the one with longer duration (more likely "real"), drop the rest.
    best = {}
    for ev in events:
        k = (ev["start"], ev["xml_string"])
        if k not in best or ev["dur"] > best[k]["dur"]:
            best[k] = ev
    events = list(best.values())
    events.sort(key=lambda e: (e["start"], e["xml_string"], e["pitch"]))

    # Group by start tick -> chords
    by_start = defaultdict(list)
    for ev in events:
        by_start[ev["start"]].append(ev)
    starts = sorted(by_start.keys())

    # MusicXML skeleton
    score = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(score, "part-list")
    sp = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(sp, "part-name").text = "Guitar TAB"
    part = etree.SubElement(score, "part", id="P1")

    measure_len = DIVISIONS * TIME_SIG_NUM

    def new_measure(num: int):
        return etree.SubElement(part, "measure", number=str(num))

    cur_measure_no = 1
    cur_measure = new_measure(cur_measure_no)

    # first measure attributes
    attrs = etree.SubElement(cur_measure, "attributes")
    etree.SubElement(attrs, "divisions").text = str(DIVISIONS)
    time = etree.SubElement(attrs, "time")
    etree.SubElement(time, "beats").text = str(TIME_SIG_NUM)
    etree.SubElement(time, "beat-type").text = str(TIME_SIG_DEN)
    add_staff_details(attrs)
    etree.SubElement(cur_measure, "sound", tempo=str(tempo_bpm))

    jams_to_note_elem = {}

    def goto_measure_for_tick(global_tick: int):
        nonlocal cur_measure_no, cur_measure
        target = (global_tick // measure_len) + 1
        while cur_measure_no < target:
            cur_measure_no += 1
            cur_measure = new_measure(cur_measure_no)

    cur_global = 0

    for idx, st in enumerate(starts):
        chord_events = by_start[st]
        goto_measure_for_tick(st)

        # forward to start time (piecewise)
        while cur_global < st:
            goto_measure_for_tick(cur_global)
            bar_start = (cur_measure_no - 1) * measure_len
            bar_end = bar_start + measure_len
            if st >= bar_end:
                delta = bar_end - cur_global
                if delta > 0:
                    fwd = etree.SubElement(cur_measure, "forward")
                    etree.SubElement(fwd, "duration").text = str(delta)
                    cur_global += delta
                cur_measure_no += 1
                cur_measure = new_measure(cur_measure_no)
            else:
                delta = st - cur_global
                if delta > 0:
                    fwd = etree.SubElement(cur_measure, "forward")
                    etree.SubElement(fwd, "duration").text = str(delta)
                    cur_global += delta

        # next onset spacing (prevents huge sustain)
        next_st = starts[idx + 1] if (idx + 1 < len(starts)) else None
        max_hold = (next_st - st) if next_st is not None else None

        # order within chord: higher string visually on top (since line 6 is high e)
        chord_events = sorted(chord_events, key=lambda e: e["xml_string"], reverse=True)

        for k, ev in enumerate(chord_events):
            note = etree.SubElement(cur_measure, "note")
            if k > 0:
                etree.SubElement(note, "chord")

            # pitch (required even for TAB in many readers)
            pitch_el = etree.SubElement(note, "pitch")
            step, alter, octv = midi_pitch_to_step_alter_oct(ev["pitch"])
            etree.SubElement(pitch_el, "step").text = step
            if alter != 0:
                etree.SubElement(pitch_el, "alter").text = str(alter)
            etree.SubElement(pitch_el, "octave").text = str(octv)

            # per-note duration, clipped to next onset
            dur = ev["dur"]
            if max_hold is not None:
                dur = min(dur, max_hold)
            dur = max(1, dur)

            etree.SubElement(note, "duration").text = str(dur)
            etree.SubElement(note, "voice").text = "1"
            etree.SubElement(note, "type").text = duration_to_type(dur)
            etree.SubElement(note, "stem").text = "none"

            notations = ensure_notations(note)
            technical = ensure_technical(notations)

            etree.SubElement(technical, "string").text = str(ev["xml_string"])
            etree.SubElement(technical, "fret").text = str(ev["fret"])

            # inline techniques (no palm mute, no bends)
            techs = ev["techs"]
            if "vibrato" in techs:
                etree.SubElement(technical, "vibrato")
            if any("harmonic" in t for t in techs):
                etree.SubElement(technical, "harmonic")
            if any(t in techs for t in ["dead", "muted", "mute"]):
                etree.SubElement(note, "notehead").text = "x"

            jams_to_note_elem[ev["j_idx"]] = note

        # advance timeline by spacing to next onset (or max dur if last)
        if max_hold is not None:
            cur_global += max_hold
        else:
            cur_global += max(e["dur"] for e in chord_events)

    # Apply spanners after notes exist
    for start_j, d in pairs.items():
        start_note = jams_to_note_elem.get(start_j)
        if start_note is None:
            continue

        if "legato" in d:
            stop_note = jams_to_note_elem.get(d["legato"])
            if stop_note is not None:
                sN = ensure_notations(start_note)
                eN = ensure_notations(stop_note)
                etree.SubElement(sN, "slur", type="start", number="1")
                etree.SubElement(eN, "slur", type="stop", number="1")

        if "slide" in d:
            stop_note = jams_to_note_elem.get(d["slide"])
            if stop_note is not None:
                sN = ensure_notations(start_note)
                eN = ensure_notations(stop_note)

                # MuseScore: a clean TAB slide is best represented as a SOLID glissando
                # with text "/", NOT "slide" (which tends to render wavy + labeled).
                g1 = etree.SubElement(sN, "glissando", attrib={
                    "type": "start",
                    "number": "1",
                    "line-type": "solid",
                })
                g1.text = "/"

                etree.SubElement(eN, "glissando", attrib={
                    "type": "stop",
                    "number": "1",
                    "line-type": "solid",
                })



    etree.ElementTree(score).write(output_xml, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"✓ Wrote {output_xml} (1/16 snapped, collisions dropped, MuseScore-friendly strings).")

def main(jam_path, output_name = "final_tab.musicxml"):
    jams_file = jam_path
    jams_to_musicxml_tab_musescore_clean(jams_file, output_xml=output_name, tempo_bpm=97)

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    jams_file = "/data/akshaj/MusicAI/Music-AI/4_Tabs/Testing/Dua Lipa - IDGAF - Cover (Fingerstyle Guitar)_colin.jams"
    jams_to_musicxml_tab_musescore_clean(jams_file, output_xml="final_tab.musicxml", tempo_bpm=97)
