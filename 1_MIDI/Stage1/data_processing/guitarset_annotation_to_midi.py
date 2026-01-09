import os
import json
import jams
import mido
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class GuitarSetJAMSConverter:
    def __init__(self, guitarset_path):
        """
        Initialize the GuitarSet JAMS to MIDI converter
        """
        self.guitarset_path = Path(guitarset_path)
        self.annotation_path = self.guitarset_path / 'annotation'
        self.midi_output_path = self.guitarset_path / 'MIDIAnnotations'

        # Create MIDI output directory (robust)
        self.midi_output_path.mkdir(parents=True, exist_ok=True)

        # MIDI timing for realistic precision:
        self.ticks_per_beat = 480     # Standard MIDI resolution
        self.default_tempo_bpm = 120  # Realistic guitar tempo   

        # Channel & program (optional): nylon guitar (24)
        self.midi_channel = 0
        self.program_number = 24

    def _sec_to_ticks(self, seconds: float) -> int:
        # Convert seconds to MIDI ticks with maximum precision
        return int(round(seconds * self.ticks_per_beat))

    def convert_jams_to_midi(self, jams_path, output_path):
        """
        Convert a single JAMS file to MIDI with maximal timing fidelity.
        """
        try:
            print(f"    Loading JAMS file: {Path(jams_path).name}")
            jam = jams.load(str(jams_path))

            # Prepare MIDI file/track
            mid = mido.MidiFile()
            mid.ticks_per_beat = self.ticks_per_beat

            track = mido.MidiTrack()
            mid.tracks.append(track)

            # Auto-detect tempo based on actual audio duration
            if notes:
                total_duration = max(start_times) + max(durations)
                # Calculate BPM that makes the MIDI duration match the audio duration
                # Assuming 4/4 time signature, calculate beats needed
                beats_needed = total_duration * 4  # 4 beats per measure
                calculated_bpm = beats_needed * 60 / total_duration
                # Clamp to reasonable range (60-200 BPM)
                calculated_bpm = max(60, min(200, calculated_bpm))
                print(f"    Auto-detected tempo: {calculated_bpm:.1f} BPM for {total_duration:.2f}s audio")
            else:
                calculated_bpm = self.default_tempo_bpm
            
            track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(calculated_bpm), time=0))
            # Time signature
            track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
            # Set instrument to guitar
            track.append(mido.Message('program_change', program=self.program_number, channel=self.midi_channel, time=0))

            # Extract all note_midi annotations (GuitarSet provides multiple blocks)
            note_annotations = jam.search(namespace='note_midi')
            print(f"    Found {len(note_annotations)} note_midi annotation blocks")
            
            if not note_annotations:
                print("    No note_midi annotations found")
                return False

            # Collect notes with exact timing from JAMS (no modifications)
            notes = []
            skipped_notes = 0
            invalid_midi_notes = 0
            zero_duration_notes = 0
            negative_time_notes = 0
            
            for ann_idx, ann in enumerate(note_annotations):
                if getattr(ann, 'data', None):
                    print(f"      Block {ann_idx + 1}: {len(ann.data)} notes")
                    for note_idx, nd in enumerate(ann.data):
                        # Convert floating point MIDI note to integer (round, don't truncate)
                        midi_note = int(round(nd.value))
                        
                        # Validate MIDI note range (0-127)
                        if midi_note < 0 or midi_note > 127:
                            print(f"        Note {note_idx}: Invalid MIDI note {midi_note} from value {nd.value}")
                            invalid_midi_notes += 1
                            continue

                        # Use EXACT time and duration from JAMS (no modifications)
                        start_s = float(nd.time)
                        dur_s = float(nd.duration)
                        
                        # Check for timing issues
                        if start_s < 0:
                            print(f"        Note {note_idx}: Negative start time {start_s}s")
                            negative_time_notes += 1
                            continue
                            
                        if dur_s <= 0.0:
                            print(f"        Note {note_idx}: Zero/negative duration {dur_s}s")
                            zero_duration_notes += 1
                            continue

                        # Use confidence if available, otherwise default velocity
                        velocity = 64
                        conf = getattr(nd, 'confidence', None)
                        if conf is not None and conf > 0:
                            velocity = max(1, min(127, int(round(conf * 127))))
                            if conf < 0.5:
                                print(f"        Note {note_idx}: Low confidence {conf:.3f} -> velocity {velocity}")

                        notes.append((start_s, dur_s, midi_note, velocity))

            if not notes:
                print("    No valid notes found")
                return False

            print(f"    Processing {len(notes)} valid notes")
            print(f"    Skipped: {skipped_notes} total, {invalid_midi_notes} invalid MIDI, {zero_duration_notes} zero duration, {negative_time_notes} negative time")
            
            # Analyze timing distribution
            if notes:
                start_times = [note[0] for note in notes]
                durations = [note[1] for note in notes]
                midi_values = [note[2] for note in notes]
                
                print(f"    Timing range: {min(start_times):.3f}s to {max(start_times):.3f}s")
                print(f"    Duration range: {min(durations):.3f}s to {max(durations):.3f}s")
                print(f"    MIDI note range: {min(midi_values)} to {max(midi_values)}")
                print(f"    Total audio duration: {max(start_times) + max(durations):.3f}s")

            # Build events with EXACT timing from JAMS
            events = []
            for start_s, dur_s, midi_note, velocity in notes:
                # Convert to ticks with maximum precision
                on_tick = self._sec_to_ticks(start_s)
                off_tick = self._sec_to_ticks(start_s + dur_s)
                
                # Ensure minimum 1 tick duration after rounding
                if off_tick <= on_tick:
                    off_tick = on_tick + 1
                    print(f"        Warning: Adjusted duration for note {midi_note} from {dur_s}s to {(off_tick - on_tick) / self.ticks_per_beat:.3f}s")

                events.append(('note_on', on_tick, midi_note, velocity))
                events.append(('note_off', off_tick, midi_note, 0))

            print(f"    Created {len(events)} MIDI events ({len(events)//2} notes)")

            # Sort by time, then by priority (note_off before note_on at same time)
            PRIORITY = {'note_off': 0, 'note_on': 1}
            events.sort(key=lambda e: (e[1], PRIORITY[e[0]]))

            # Emit events with proper delta times (maintaining exact timing)
            prev_time = 0
            max_delta = 0
            min_delta = float('inf')
            total_delta = 0
            
            for etype, abs_time, note, vel in events:
                delta = abs_time - prev_time
                
                # Track delta time statistics
                if delta > max_delta:
                    max_delta = delta
                if delta < min_delta:
                    min_delta = delta
                total_delta += delta
                
                # Create MIDI message with exact timing
                if etype == 'note_on':
                    msg = mido.Message('note_on', channel=self.midi_channel, note=note, velocity=vel, time=delta)
                else:
                    msg = mido.Message('note_off', channel=self.midi_channel, note=note, velocity=vel, time=delta)
                
                track.append(msg)
                prev_time = abs_time

            # Print timing statistics
            if events:
                avg_delta = total_delta / len(events)
                print(f"    MIDI timing stats: min={min_delta}, max={max_delta}, avg={avg_delta:.1f} ticks")
                print(f"    Time resolution: {1/self.ticks_per_beat*60:.3f}s per tick at {calculated_bpm:.1f} BPM")

            track.append(mido.MetaMessage('end_of_track', time=0))
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            mid.save(str(output_path))

            print(f"    ✓ Saved MIDI: {Path(output_path).name} (notes: {len(notes)})")
            return True

        except Exception as e:
            print(f"    ✗ Error converting {Path(jams_path).name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_all_annotations(self):
        """
        Process all JAMS annotation files in the GuitarSet dataset
        """
        print("Starting GuitarSet JAMS to MIDI conversion...")
        print("=" * 60)

        if not self.annotation_path.exists():
            print(f"Annotation directory not found at: {self.annotation_path}")
            return

        jams_files = list(self.annotation_path.glob('*.jams'))
        if not jams_files:
            print("No JAMS files found in annotation directory")
            return

        print(f"Found {len(jams_files)} JAMS annotation files")
        print(f"Converting to MIDI and saving to: {self.midi_output_path}")
        print("=" * 60)

        ok = 0
        fail = 0
        for jams_file in jams_files:
            print(f"Processing: {jams_file.name}")
            midi_output_path = self.midi_output_path / f"{jams_file.stem}.mid"
            if self.convert_jams_to_midi(jams_file, midi_output_path):
                ok += 1
            else:
                fail += 1

        print("=" * 60)
        print("CONVERSION COMPLETE!")
        print(f"Successful conversions: {ok}")
        print(f"Failed conversions: {fail}")
        print(f"Total files processed: {len(jams_files)}")
        print(f"MIDI files saved to: {self.midi_output_path}")
        print("=" * 60)


def main():
    """Main execution function"""
    guitarset_path = "/data/akshaj/MusicAI/GuitarSet"
    if not os.path.exists(guitarset_path):
        print(f"GuitarSet dataset not found at: {guitarset_path}")
        print("Please check the path and try again.")
        return

    annotation_path = os.path.join(guitarset_path, 'annotation')
    if not os.path.exists(annotation_path):

        return

    converter = GuitarSetJAMSConverter(guitarset_path)
    converter.process_all_annotations()


if __name__ == "__main__":
    main()