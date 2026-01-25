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

# MuseScore TAB: staff-tuning line=1 is LOWEST string (E2).
STAFF_TUNING_LINE_1_TO_6_LOW_TO_HIGH = [
    ("E", 2), ("A", 2), ("D", 3), ("G", 3), ("B", 3), ("E", 4)
]

# If your JAMS uses 1=LOW E ... 6=HIGH e, keep False.
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
    qn = seconds_to_quarters(sec, tempo_bpm)
    ticks = int(round(qn * DIVISIONS))
    return int(round(ticks / GRID_TICKS)) * GRID_TICKS

def quantize_dur_to_grid_ticks(sec, tempo_bpm):
    qn = seconds_to_quarters(sec, tempo_bpm)
    ticks = int(round(qn * DIVISIONS))
    qt = int(round(ticks / GRID_TICKS)) * GRID_TICKS
    return max(GRID_TICKS, qt)

def seconds_to_ticks(sec, tempo_bpm):
    qn = seconds_to_quarters(sec, tempo_bpm)
    return int(round(qn * DIVISIONS))

def cluster_onsets(tab_data, tempo_bpm, cluster_ms=30.0):
    """
    dict[jams_index] = clustered_start_tick
    Cluster notes whose onsets are within cluster_ms, quantize once per cluster.
    """
    items = [(i, float(obs.time)) for i, obs in enumerate(tab_data)]
    items.sort(key=lambda x: x[1])
    if not items:
        return {}

    cluster_ticks = seconds_to_ticks(cluster_ms / 1000.0, tempo_bpm)

    clusters = []
    cur = [items[0]]
    for idx, t in items[1:]:
        prev_t = cur[-1][1]
        if seconds_to_ticks(t - prev_t, tempo_bpm) <= cluster_ticks:
            cur.append((idx, t))
        else:
            clusters.append(cur)
            cur = [(idx, t)]
    clusters.append(cur)

    idx_to_start = {}
    for cl in clusters:
        times = sorted(t for _, t in cl)
        med_t = times[len(times)//2]
        q_start = quantize_to_grid_ticks(med_t, tempo_bpm)
        for idx, _ in cl:
            idx_to_start[idx] = q_start

    return idx_to_start

def duration_to_type(dur_ticks):
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

def jams_string_to_xml_string(jams_string: int) -> int:
    s = int(jams_string)
    return (7 - s) if JAMS_IS_1_HIGH else s


# ----------------------------
# Attributes writers
# ----------------------------
def add_time_and_divisions(attrs_elem):
    etree.SubElement(attrs_elem, "divisions").text = str(DIVISIONS)
    time = etree.SubElement(attrs_elem, "time")
    etree.SubElement(time, "beats").text = str(TIME_SIG_NUM)
    etree.SubElement(time, "beat-type").text = str(TIME_SIG_DEN)

def add_standard_clef(attrs_elem):
    clef = etree.SubElement(attrs_elem, "clef")
    etree.SubElement(clef, "sign").text = "G"
    etree.SubElement(clef, "line").text = "2"

def add_tab_staff_details(attrs_elem):
    clef = etree.SubElement(attrs_elem, "clef")
    etree.SubElement(clef, "sign").text = "TAB"
    etree.SubElement(clef, "line").text = "5"

    sd = etree.SubElement(attrs_elem, "staff-details")
    etree.SubElement(sd, "staff-lines").text = "6"

    for line, (step, octave) in enumerate(STAFF_TUNING_LINE_1_TO_6_LOW_TO_HIGH, start=1):
        st = etree.SubElement(sd, "staff-tuning", line=str(line))
        etree.SubElement(st, "tuning-step").text = step
        etree.SubElement(st, "tuning-octave").text = str(octave)


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
            # ignore percussive / non-melodic tags for spanners
            techs.discard("palm_muting")
            techs.discard("snare_drum")

            if not techs:
                continue

            fa = oa.value.get("fret", None)
            fb = ob.value.get("fret", None)
            if fa is not None and fb is not None:
                if abs(int(fb) - int(fa)) > max_fret_jump:
                    continue

            if ("hammer_on_pull_off" in techs) or ("hopo" in techs) or ("hammer" in techs) or ("pull" in techs):
                pairs.setdefault(a, {})["legato"] = b

            if "slide" in techs:
                pairs.setdefault(a, {})["slide"] = b

    return pairs


# ----------------------------
# Main writer
# ----------------------------
def jams_to_musicxml_standard_plus_tab_TWO_PARTS(
    jams_path,
    output_xml="final_standard_plus_tab.musicxml",
    tempo_bpm=120,
    pair_window_s=PAIR_WINDOW_S,
):
    jam = jams.load(jams_path, validate=False)
    tab_ann = next((a for a in jam.annotations if a.namespace == "tab_note"), None)
    if tab_ann is None:
        raise ValueError("No tab_note annotation found (namespace='tab_note').")

    tab_data = sorted(tab_ann.data, key=lambda o: (float(o.time), int(o.value["string"])))
    if not tab_data:
        raise ValueError("tab_note annotation is empty.")

    pairs = build_pairs(tab_data, max_dt=pair_window_s)
    start_map = cluster_onsets(tab_data, tempo_bpm, cluster_ms=30.0)

    # Build events (single unified timeline)
    events = []
    for j_idx, obs in enumerate(tab_data):
        v = obs.value
        pitch = int(v["pitch"])
        xml_string = jams_string_to_xml_string(int(v["string"]))
        fret = int(v["fret"]) if ("fret" in v and v["fret"] is not None) else 0

        start = start_map[j_idx]
        dur = quantize_dur_to_grid_ticks(float(obs.duration), tempo_bpm)

        techs = set(normalize_techs(v))
        # drop percussive tags from note-marking too (optional)
        techs.discard("palm_muting")
        techs.discard("snare_drum")

        events.append(dict(
            j_idx=j_idx,
            start=start,
            dur=dur,
            pitch=pitch,
            xml_string=xml_string,
            fret=fret,
            techs=techs
        ))

    # Collision removal (same start,string): keep longer dur
    best = {}
    for ev in events:
        k = (ev["start"], ev["xml_string"])
        if k not in best or ev["dur"] > best[k]["dur"]:
            best[k] = ev
    events = list(best.values())
    events.sort(key=lambda e: (e["start"], e["xml_string"], e["pitch"]))

    # Group by onset -> chords
    by_start = defaultdict(list)
    for ev in events:
        by_start[ev["start"]].append(ev)
    starts = sorted(by_start.keys())

    # ----------------------------
    # MusicXML skeleton: TWO PARTS
    # ----------------------------
    score = etree.Element("score-partwise", version="4.0")
    part_list = etree.SubElement(score, "part-list")

    # P1 = standard
    sp1 = etree.SubElement(part_list, "score-part", id="P1")
    etree.SubElement(sp1, "part-name").text = "Guitar (Standard)"
    part_std = etree.SubElement(score, "part", id="P1")

    # P2 = tab
    sp2 = etree.SubElement(part_list, "score-part", id="P2")
    etree.SubElement(sp2, "part-name").text = "Guitar TAB"
    part_tab = etree.SubElement(score, "part", id="P2")

    measure_len = DIVISIONS * TIME_SIG_NUM

    def new_measure(part, num: int):
        return etree.SubElement(part, "measure", number=str(num))

    cur_measure_no = 1
    cur_meas_std = new_measure(part_std, cur_measure_no)
    cur_meas_tab = new_measure(part_tab, cur_measure_no)

    # First-measure attributes for BOTH parts
    attrs_std = etree.SubElement(cur_meas_std, "attributes")
    add_time_and_divisions(attrs_std)
    add_standard_clef(attrs_std)
    direction = etree.SubElement(cur_meas_std, "direction", placement="above")
    direction_type = etree.SubElement(direction, "direction-type")

    met = etree.SubElement(direction_type, "metronome")
    etree.SubElement(met, "beat-unit").text = "quarter"
    etree.SubElement(met, "per-minute").text = str(tempo_bpm)

    # keep playback tempo too
    etree.SubElement(direction, "sound", tempo=str(tempo_bpm))


    attrs_tab = etree.SubElement(cur_meas_tab, "attributes")
    add_time_and_divisions(attrs_tab)
    add_tab_staff_details(attrs_tab)
    etree.SubElement(cur_meas_tab, "sound", tempo=str(tempo_bpm))

    # For spanners we only want TAB note elements (MuseScore interprets those well)
    jams_to_note_elem_tab = {}

    def goto_measure_for_tick(global_tick: int):
        nonlocal cur_measure_no, cur_meas_std, cur_meas_tab
        target = (global_tick // measure_len) + 1
        while cur_measure_no < target:
            cur_measure_no += 1
            cur_meas_std = new_measure(part_std, cur_measure_no)
            cur_meas_tab = new_measure(part_tab, cur_measure_no)

    # Explicit rests in BOTH parts (so standard staff matches gaps)
    cur_global = 0

    def add_rest_both(duration_ticks: int):
        nonlocal cur_measure_no, cur_meas_std, cur_meas_tab, cur_global

        remaining = duration_ticks
        while remaining > 0:
            bar_start = (cur_measure_no - 1) * measure_len
            bar_end = bar_start + measure_len
            cur_pos_in_bar = cur_global - bar_start
            room = bar_end - (bar_start + cur_pos_in_bar)
            take = min(remaining, room)

            # standard rest
            r1 = etree.SubElement(cur_meas_std, "note")
            etree.SubElement(r1, "rest")
            etree.SubElement(r1, "duration").text = str(take)
            etree.SubElement(r1, "voice").text = "1"
            etree.SubElement(r1, "type").text = duration_to_type(take)
            etree.SubElement(r1, "stem").text = "none"

            # tab rest
            r2 = etree.SubElement(cur_meas_tab, "note")
            etree.SubElement(r2, "rest")
            etree.SubElement(r2, "duration").text = str(take)
            etree.SubElement(r2, "voice").text = "1"
            etree.SubElement(r2, "type").text = duration_to_type(take)
            etree.SubElement(r2, "stem").text = "none"

            cur_global += take
            remaining -= take

            if remaining > 0:
                # advance to next measure in BOTH parts
                cur_measure_no += 1
                cur_meas_std = new_measure(part_std, cur_measure_no)
                cur_meas_tab = new_measure(part_tab, cur_measure_no)

    # ----------------------------
    # Emit notes into BOTH parts
    # ----------------------------
    for idx, st in enumerate(starts):
        chord_events = by_start[st]
        goto_measure_for_tick(st)

        # Fill gap with explicit rests (fixes drums-only pauses)
        if cur_global < st:
            add_rest_both(st - cur_global)

        # Next onset spacing cap
        next_st = starts[idx + 1] if (idx + 1 < len(starts)) else None
        max_hold = (next_st - st) if next_st is not None else None

        # Sort chord events: higher string on top in TAB
        chord_events = sorted(chord_events, key=lambda e: e["xml_string"], reverse=True)

        # ----- STANDARD STAFF (P1) -----
        for k, ev in enumerate(chord_events):
            n = etree.SubElement(cur_meas_std, "note")
            if k > 0:
                etree.SubElement(n, "chord")

            pitch_el = etree.SubElement(n, "pitch")
            step, alter, octv = midi_pitch_to_step_alter_oct(ev["pitch"])
            etree.SubElement(pitch_el, "step").text = step
            if alter != 0:
                etree.SubElement(pitch_el, "alter").text = str(alter)
            etree.SubElement(pitch_el, "octave").text = str(octv)

            dur = ev["dur"]
            if max_hold is not None:
                dur = min(dur, max_hold)
            dur = max(1, dur)

            etree.SubElement(n, "duration").text = str(dur)
            etree.SubElement(n, "voice").text = "1"
            etree.SubElement(n, "type").text = duration_to_type(dur)
            etree.SubElement(n, "stem").text = "up"

        # ----- TAB STAFF (P2) -----
        for k, ev in enumerate(chord_events):
            note = etree.SubElement(cur_meas_tab, "note")
            if k > 0:
                etree.SubElement(note, "chord")

            # Keep pitch too (helps importers)
            pitch_el = etree.SubElement(note, "pitch")
            step, alter, octv = midi_pitch_to_step_alter_oct(ev["pitch"])
            etree.SubElement(pitch_el, "step").text = step
            if alter != 0:
                etree.SubElement(pitch_el, "alter").text = str(alter)
            etree.SubElement(pitch_el, "octave").text = str(octv)

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

            techs = ev["techs"]

            # vibrato
            if "vibrato" in techs:
                etree.SubElement(technical, "vibrato")

            # harmonics: DO NOT use <harmonic/> (MuseScore draws the circle)
            # Use fingering text (what you showed as desired).
            if any("harmonic" in t for t in techs):
                fing = etree.SubElement(technical, "fingering")
                fing.text = "Harm."

            # dead/muted notes
            if any(t in techs for t in ["dead", "muted", "mute"]):
                etree.SubElement(note, "notehead").text = "x"

            # bends (with bend points)
            if "bend" in techs:
                bend = etree.SubElement(technical, "bend")
                etree.SubElement(bend, "bend-alter").text = "1"
                etree.SubElement(bend, "bend-point", attrib={"time": "0", "alter": "0"})
                etree.SubElement(bend, "bend-point", attrib={"time": "50", "alter": "1"})
                etree.SubElement(bend, "bend-point", attrib={"time": "100", "alter": "1"})
                # fallback text
                direction = etree.SubElement(note, "direction", placement="above")
                direction_type = etree.SubElement(direction, "direction-type")
                words = etree.SubElement(direction_type, "words")
                words.text = "bend"

            jams_to_note_elem_tab[ev["j_idx"]] = note

        # Advance time to next onset (so rests can exist)
        if max_hold is not None:
            cur_global += max_hold
        else:
            cur_global += max(e["dur"] for e in chord_events)

    # ----------------------------
    # Spanners on TAB part only
    # ----------------------------
    for start_j, d in pairs.items():
        start_note = jams_to_note_elem_tab.get(start_j)
        if start_note is None:
            continue

        if "legato" in d:
            stop_note = jams_to_note_elem_tab.get(d["legato"])
            if stop_note is not None:
                sN = ensure_notations(start_note)
                eN = ensure_notations(stop_note)
                etree.SubElement(sN, "slur", type="start", number="1")
                etree.SubElement(eN, "slur", type="stop", number="1")

        if "slide" in d:
            stop_note = jams_to_note_elem_tab.get(d["slide"])
            if stop_note is not None:
                sN = ensure_notations(start_note)
                eN = ensure_notations(stop_note)
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
    print(f"✓ Wrote {output_xml} (TWO PARTS: standard+TAB; TAB preserved, no re-fingering).")

def main(jam_path, output_name = "final_tab.musicxml", bpm=120):
    jams_file = jam_path
    jams_to_musicxml_standard_plus_tab_TWO_PARTS(jams_file, output_xml=output_name, tempo_bpm=bpm)

# ----------------------------
# RUN
# ----------------------------
# if __name__ == "__main__":
#     jams_file = "/data/akshaj/MusicAI/Music-AI/4_Tabs/Testing/Dua Lipa - IDGAF - Cover (Fingerstyle Guitar)_colin.jams"
#     jams_to_musicxml_tab_musescore_clean(jams_file, output_xml="final_tab.musicxml", tempo_bpm=97)
