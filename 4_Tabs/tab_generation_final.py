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

# CHANGE: use beat-based window instead of seconds (consistent across BPMs)
PAIR_WINDOW_BEATS = 1.0       # 1 beat = quarter note at tempo_bpm

# MuseScore TAB: staff-tuning line=1 is LOWEST string (E2).
STAFF_TUNING_LINE_1_TO_6_LOW_TO_HIGH = [
    ("E", 2), ("A", 2), ("D", 3), ("G", 3), ("B", 3), ("E", 4)
]

# If your JAMS uses 1=LOW E ... 6=HIGH e, keep False.
JAMS_IS_1_HIGH = False


# ----------------------------
# Helpers
# ----------------------------
def beats_to_seconds(beats: float, tempo_bpm: float) -> float:
    # 1 beat = a quarter note at tempo_bpm
    return float(beats) * (60.0 / float(tempo_bpm))

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
def build_pairs(tab_data, tempo_bpm, max_dt_beats=1.0, max_fret_jump=7):
    """
    CHANGE: pairing window is in beats (quarter notes). We convert beats->seconds using tempo_bpm
    so the same musical window behaves consistently across BPMs.
    """
    max_dt = beats_to_seconds(max_dt_beats, tempo_bpm)

    by_string = defaultdict(list)
    for i, obs in enumerate(tab_data):
        by_string[int(obs.value["string"])].append(i)

    # IMPORTANT: ensure indices are time-ordered within each string
    for s in by_string:
        by_string[s].sort(key=lambda idx: float(tab_data[idx].time))

    pairs = {}
    for s, idxs in by_string.items():
        for a, b in zip(idxs, idxs[1:]):
            oa, ob = tab_data[a], tab_data[b]
            dt = float(ob.time) - float(oa.time)
            if dt <= 0 or dt > max_dt:
                continue

            techs = set(normalize_techs(oa.value))
            techs.discard("palm_muting")
            techs.discard("snare_drum")

            if not techs:
                continue

            fa = oa.value.get("fret", None)
            fb = ob.value.get("fret", None)
            if fa is None or fb is None:
                continue

            fa = int(fa); fb = int(fb)

            # HARD GUARDS:
            # 1) Same string is already guaranteed by grouping
            # 2) Must actually change fret (no 0->0, 2->2)
            if fa == fb:
                continue

            # 3) avoid insane jumps
            if abs(fb - fa) > max_fret_jump:
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
    pair_window_beats=PAIR_WINDOW_BEATS,   # CHANGE: beats, not seconds
):
    jam = jams.load(jams_path, validate=False)
    tab_ann = next((a for a in jam.annotations if a.namespace == "tab_note"), None)
    if tab_ann is None:
        raise ValueError("No tab_note annotation found (namespace='tab_note').")

    tab_data = sorted(tab_ann.data, key=lambda o: (float(o.time), int(o.value["string"])))
    if not tab_data:
        raise ValueError("tab_note annotation is empty.")

    # CHANGE: build_pairs now takes tempo_bpm + beat window
    pairs = build_pairs(tab_data, tempo_bpm, max_dt_beats=pair_window_beats)
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

    jidx_to_start = {ev["j_idx"]: ev["start"] for ev in events}

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
    note_meta = {}  # j_idx -> (xml_string, fret) for spanner validation

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
            note_meta[ev["j_idx"]] = (ev["xml_string"], ev["fret"])

        # Advance time to next onset (so rests can exist)
        if max_hold is not None:
            cur_global += max_hold
        else:
            cur_global += max(e["dur"] for e in chord_events)

    # ----------------------------
    # Spanners on TAB part only (MuseScore-stable: no overlaps share numbers)
    # ----------------------------
    def allocate_numbers_for_intervals(intervals, max_pool=16):
        """
        intervals: list of dicts with keys: s_t, e_t, start_j, stop_j
        Returns: list of (start_j, stop_j, num) with num assigned so overlapping intervals never share num.
        """
        intervals = sorted(intervals, key=lambda x: (x["s_t"], x["e_t"]))

        active = []  # list of (end_t, num)
        free_nums = list(range(1, max_pool + 1))
        out = []

        for it in intervals:
            s_t, e_t = it["s_t"], it["e_t"]

            still_active = []
            for end_t, num in active:
                if end_t <= s_t:
                    free_nums.append(num)
                else:
                    still_active.append((end_t, num))
            active = still_active
            free_nums.sort()

            if not free_nums:
                max_pool += 8
                free_nums.extend(range(len(set(n for _, n in active)) + 1, max_pool + 1))
                free_nums.sort()

            num = free_nums.pop(0)
            active.append((e_t, num))
            out.append((it["start_j"], it["stop_j"], str(num)))

        return out

    # Build legato/slide intervals from pairs (with strict guards)
    legato_intervals = []
    slide_intervals = []

    for start_j, d in pairs.items():
        s_note = jams_to_note_elem_tab.get(start_j)
        s_meta = note_meta.get(start_j)
        s_t = jidx_to_start.get(start_j)

        if (s_note is None) or (s_meta is None) or (s_t is None):
            continue
        s_string, s_fret = s_meta

        # LEGATO interval candidate
        if "legato" in d:
            stop_j = d["legato"]
            e_note = jams_to_note_elem_tab.get(stop_j)
            e_meta = note_meta.get(stop_j)
            e_t = jidx_to_start.get(stop_j)

            if (e_note is not None) and (e_meta is not None) and (e_t is not None) and (e_t > s_t):
                e_string, e_fret = e_meta
                if (s_string == e_string) and (int(s_fret) != int(e_fret)):
                    legato_intervals.append({"s_t": s_t, "e_t": e_t, "start_j": start_j, "stop_j": stop_j})

        # SLIDE interval candidate
        if "slide" in d:
            stop_j = d["slide"]
            e_note = jams_to_note_elem_tab.get(stop_j)
            e_meta = note_meta.get(stop_j)
            e_t = jidx_to_start.get(stop_j)

            if (e_note is not None) and (e_meta is not None) and (e_t is not None) and (e_t > s_t):
                e_string, e_fret = e_meta
                if (s_string == e_string) and (int(s_fret) != int(e_fret)):
                    slide_intervals.append({"s_t": s_t, "e_t": e_t, "start_j": start_j, "stop_j": stop_j})

    # Allocate numbers so overlaps never share the same one
    legato_assigned = allocate_numbers_for_intervals(legato_intervals, max_pool=16)
    slide_assigned  = allocate_numbers_for_intervals(slide_intervals,  max_pool=16)

    # Emit LEGATO
    for start_j, stop_j, num in legato_assigned:
        start_note = jams_to_note_elem_tab.get(start_j)
        stop_note  = jams_to_note_elem_tab.get(stop_j)
        if start_note is None or stop_note is None:
            continue
        sN = ensure_notations(start_note)
        eN = ensure_notations(stop_note)
        etree.SubElement(sN, "slur", type="start", number=num)
        etree.SubElement(eN, "slur", type="stop",  number=num)

    # Emit SLIDE
    for start_j, stop_j, num in slide_assigned:
        start_note = jams_to_note_elem_tab.get(start_j)
        stop_note  = jams_to_note_elem_tab.get(stop_j)
        if start_note is None or stop_note is None:
            continue
        sN = ensure_notations(start_note)
        eN = ensure_notations(stop_note)
        g1 = etree.SubElement(sN, "glissando", attrib={
            "type": "start",
            "number": num,
            "line-type": "solid",
        })
        g1.text = "/"
        etree.SubElement(eN, "glissando", attrib={
            "type": "stop",
            "number": num,
            "line-type": "solid",
        })

    etree.ElementTree(score).write(output_xml, pretty_print=True, xml_declaration=True, encoding="UTF-8")
    print(f"✓ Wrote {output_xml} (TWO PARTS: standard+TAB; TAB preserved, no re-fingering).")


def main(jam_path, output_name = "final_tab.musicxml", bpm=120):
    jams_file = jam_path
    jams_to_musicxml_standard_plus_tab_TWO_PARTS(jams_file, output_xml=output_name, tempo_bpm=bpm, pair_window_beats=1.0)


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    jams_file = "/data/shamakg/music_ai_pipeline/Music-AI/5_FinalPipeline/Dua Lipa - IDGAF - Cover (Fingerstyle Guitar).jams"
    jams_to_musicxml_standard_plus_tab_TWO_PARTS(
        jams_file,
        output_xml="final_standard_plus_tab.musicxml",
        tempo_bpm=97,
        pair_window_beats=1.0,   # 1-beat legato/slide pairing window
    )

