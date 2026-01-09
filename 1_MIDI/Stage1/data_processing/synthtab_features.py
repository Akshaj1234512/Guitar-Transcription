import io
import os
import logging
import h5py
import librosa
import pretty_midi
import numpy as np
import tempfile
import zipfile
import shutil
from boxsdk import Client, OAuth2

# --- CONFIGURATION ---
DEVELOPER_TOKEN = 'qtxKUpclyz6siRatBPIIz6mk3p9V5REK' 
SHARED_LINK = "https://rochester.app.box.com/v/SynthTab-Full"

# Local paths
OUTPUT_DIR = "/data/akshaj/MusicAI/workspace/hdf5s/synthtab/2024"
TEMP_DIR = "/data/akshaj/MusicAI/temp_cache" 

SAMPLE_RATE = 16000 

# Setup Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conversion_log.txt"),
        logging.StreamHandler()
    ]
)

def get_box_client(token):
    auth = OAuth2(client_id='', client_secret='', access_token=token)
    return Client(auth)

def float32_to_int16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.).astype(np.int16)

def normalize_audio(audio):
    return audio / (np.max(np.abs(audio)) + 1e-9)

# --- NEW: MERGE 6 STRING MIDIS INTO ONE ---
def merge_and_parse_midi(midi_file_list):
    """
    Takes a list of paths ['.../string_1.mid', '.../string_6.mid']
    Merges them into one event dictionary.
    """
    try:
        midi_dict = {'midi_event': [], 'midi_event_time': []}
        
        for midi_path in midi_file_list:
            try:
                # Extract string number from filename if possible (string_6.mid)
                base = os.path.basename(midi_path)
                string_num = 0
                if "string_" in base:
                    try:
                        string_num = int(base.replace("string_", "").replace(".mid", ""))
                    except: pass

                pm = pretty_midi.PrettyMIDI(midi_path)
                for instrument in pm.instruments:
                    for note in instrument.notes:
                        # We add the string number to the event text for clarity
                        on_str = f"note_on channel=0 note={note.pitch} velocity={note.velocity} time={note.start:.4f} string={string_num}"
                        midi_dict['midi_event'].append(on_str)
                        midi_dict['midi_event_time'].append(note.start)
                        
                        off_str = f"note_off channel=0 note={note.pitch} velocity=0 time={note.end:.4f} string={string_num}"
                        midi_dict['midi_event'].append(off_str)
                        midi_dict['midi_event_time'].append(note.end)
            except Exception as e:
                logging.warning(f"Skipped corrupt midi string file {midi_path}: {e}")

        # Sort all combined events by time
        if len(midi_dict['midi_event_time']) > 0:
            sorted_indices = np.argsort(midi_dict['midi_event_time'])
            midi_dict['midi_event'] = [midi_dict['midi_event'][i] for i in sorted_indices]
            midi_dict['midi_event_time'] = [midi_dict['midi_event_time'][i] for i in sorted_indices]
            
        return midi_dict
    except Exception as e:
        logging.error(f"Error merging MIDIs: {e}")
        return None

def create_h5_file(output_path, audio_data, midi_dict, meta):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as hf:
            hf.attrs.create("dataset", data="synthtab".encode(), dtype="S20")
            hf.attrs.create("split", data=meta['split'].encode(), dtype="S20")
            hf.attrs.create("audio_type", data=meta['audio_type'].encode(), dtype="S50")
            hf.attrs.create("item_id", data=meta['item_id'].encode(), dtype="S100") 
            hf.attrs.create("audio_filename", data=meta['audio_name'].encode(), dtype="S200")
            hf.attrs.create("duration", data=meta['duration'], dtype=np.float32)
            
            hf.create_dataset("waveform", data=float32_to_int16(audio_data), dtype=np.int16)
            hf.create_dataset("midi_event", data=[e.encode('utf-8') for e in midi_dict["midi_event"]], dtype="S150")
            hf.create_dataset("midi_event_time", data=midi_dict["midi_event_time"], dtype=np.float32)
        return True
    except Exception as e:
        logging.error(f"Failed to write H5 {output_path}: {e}")
        if os.path.exists(output_path): os.remove(output_path)
        return False

# --- NEW: CLEAN FOLDER NAMES FOR MATCHING ---
def clean_folder_name(name):
    """
    Removes suffixes like '__midi', '__audio', ' (2)' to find the core song ID.
    """
    name = name.lower()
    name = name.replace("__midi", "").replace("__audio", "")
    return name.strip()

def setup_midi_cache(root_folder):
    midi_cache_dir = os.path.join(TEMP_DIR, "midi_cache")
    if not os.path.exists(midi_cache_dir):
        os.makedirs(midi_cache_dir)
    
    # Check if we have files
    has_files = False
    for _, _, files in os.walk(midi_cache_dir):
        if any(f.endswith('.mid') for f in files):
            has_files = True
            break
    
    if has_files:
        logging.info("MIDI cache found.")
        return index_midi_folders(midi_cache_dir)

    logging.info("Searching for MIDI zip in Box root...")
    midi_item = None
    for item in root_folder.get_items():
        if item.name == 'all_jams_midi_V2_60000_tracks.zip' or ('midi' in item.name.lower() and item.name.endswith('.zip')):
            midi_item = item
            if item.name == 'all_jams_midi_V2_60000_tracks.zip': break 
    
    if not midi_item:
        raise Exception("Could not find the global MIDI zip file.")

    logging.info(f"Downloading MIDI Pack: {midi_item.name}...")
    zip_path = os.path.join(TEMP_DIR, midi_item.name)
    with open(zip_path, 'wb') as f:
        midi_item.download_to(f)
    
    logging.info("Extracting MIDI zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(midi_cache_dir)
    os.remove(zip_path)
    
    return index_midi_folders(midi_cache_dir)

# --- NEW: INDEX BY FOLDER, NOT FILE ---
def index_midi_folders(cache_dir):
    logging.info("Indexing MIDI Folders...")
    # Dictionary: { "cleaned_folder_name": ["path/to/string_1.mid", "path/to/string_2.mid"] }
    index = {}
    
    for root, dirs, files in os.walk(cache_dir):
        midis = [os.path.join(root, f) for f in files if f.endswith('.mid')]
        if midis:
            folder_name = os.path.basename(root)
            key = clean_folder_name(folder_name)
            index[key] = midis
            
    logging.info(f"Indexed {len(index)} Unique MIDI Song Folders.")
    return index

def process_audio_zip(zip_item, midi_index, path_prefix, train_ratio=0.8):
    zip_name = zip_item.name
    
    # Check Completed Marker
    marker_path = os.path.join(TEMP_DIR, f"{zip_name}.COMPLETED")
    if os.path.exists(marker_path):
        logging.info(f"Skipping {zip_name} (Completed).")
        return

    logging.info(f"--- Processing Audio Zip: {zip_name} ---")
    local_zip = os.path.join(TEMP_DIR, zip_name)
    extract_dir = os.path.join(TEMP_DIR, "audio_extract", os.path.splitext(zip_name)[0])
    
    # 1. DOWNLOAD
    need_download = True
    if os.path.exists(local_zip):
        try:
            with zipfile.ZipFile(local_zip, 'r') as z: 
                if z.testzip() is None: need_download = False
        except: pass
        
    if need_download:
        try:
            with open(local_zip, 'wb') as f: zip_item.download_to(f)
        except Exception as e:
            logging.error(f"Download failed: {e}")
            return

    # 2. EXTRACT
    if not os.path.exists(extract_dir):
        try:
            logging.info("Extracting...")
            with zipfile.ZipFile(local_zip, 'r') as z: z.extractall(extract_dir)
            os.remove(local_zip) # Save space
        except Exception as e:
            logging.error(f"Extraction failed: {e}")
            return

    # 3. CONVERT WITH FOLDER MATCHING
    try:
        count = 0
        
        # We traverse the extracted audio directories
        for root, dirs, files in os.walk(extract_dir):
            audio_files = [f for f in files if f.lower().endswith(('.wav', '.flac', '.mp3'))]
            if not audio_files: continue

            # MATCHING: Look at the current folder name
            current_folder = os.path.basename(root)
            match_key = clean_folder_name(current_folder)
            
            # Find corresponding MIDI files
            midi_files = midi_index.get(match_key)
            
            # If no exact match, try fuzzy (sometimes spacing differs)
            if not midi_files:
                # Fallback: Check if key is contained in any index key
                pass 

            if not midi_files:
                # Log only once per folder to avoid spam
                # logging.warning(f"No MIDI match for audio folder: {current_folder}")
                continue

            # Process every audio file in this matched folder
            for f in audio_files:
                clean_type = path_prefix.replace("/", "_").strip("_").lower()
                item_id = f"{clean_type}__{match_key}__{f}"
                
                is_train = (hash(item_id) % 100) < (train_ratio * 100)
                split = "train" if is_train else "validation"
                output_path = os.path.join(OUTPUT_DIR, split, f"{item_id}.h5")

                if os.path.exists(output_path): continue

                try:
                    audio_path = os.path.join(root, f)
                    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                    y = normalize_audio(y)
                    
                    # Merge the MIDIs (Parse ONLY once per folder effectively, but here we do per file)
                    midi_dict = merge_and_parse_midi(midi_files)
                    
                    if midi_dict and create_h5_file(output_path, y, midi_dict, {
                        'split': split,
                        'audio_name': f,
                        'midi_name': "merged_strings",
                        'duration': len(y)/sr,
                        'item_id': item_id,
                        'audio_type': clean_type
                    }):
                        count += 1
                except Exception as e:
                    logging.error(f"Error {f}: {e}")

        logging.info(f"Finished {zip_name}: Created {count} H5 files.")
        
        # Success Marker
        with open(marker_path, 'w') as f: f.write("done")

    finally:
        if os.path.exists(extract_dir): shutil.rmtree(extract_dir)

def traverse_box(folder, path_prefix, midi_index):
    try:
        items = folder.get_items()
        for item in items:
            if item.type == 'folder':
                traverse_box(item, f"{path_prefix}_{item.name}", midi_index)
            elif item.type == 'file' and item.name.endswith('.zip') and 'midi' not in item.name.lower():
                process_audio_zip(item, midi_index, path_prefix)
    except Exception as e:
        logging.error(f"Error traversing folder: {e}")

def main():
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
    try:
        client = get_box_client(DEVELOPER_TOKEN)
        root_folder = client.get_shared_item(SHARED_LINK)
        
        logging.info("Phase 1: Setting up MIDI Cache...")
        midi_index = setup_midi_cache(root_folder)
        
        logging.info("Phase 2: Processing Audio Zips...")
        traverse_box(root_folder, "synthtab", midi_index)
    except Exception as e:
        logging.critical(f"Script failed: {e}")

if __name__ == "__main__":
    main()