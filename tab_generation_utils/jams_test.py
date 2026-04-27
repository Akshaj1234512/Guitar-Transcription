

import jams
import numpy as np
import os
import pretty_midi
from music21 import stream, note, tempo, meter, clef, articulations, expressions, spanner, interval, instrument, metadata
from music21 import note as m21_note  # Ensure we have access to Rest
from music21 import chord as m21_chord
import xml.etree.ElementTree as ET

import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import math

def save_jam(jam, output_name):
    try:
        # Calculate duration from annotations if not set
        if jam.file_metadata.duration is None or jam.file_metadata.duration <= 0:
            max_time = 0
            for ann in jam.annotations:
                if len(ann.data) > 0:
                    last_obs = ann.data[-1]
                    end_time = last_obs.time + getattr(last_obs, 'duration', 0)
                    max_time = max(max_time, end_time)
            jam.file_metadata.duration = max_time if max_time > 0 else 1.0
            print(f"Set JAMS duration to {jam.file_metadata.duration:.2f}s")
        
        jams_output_path = output_name.replace('.xml', '.jams')
        
        # Manually construct the JSON dict to bypass validation
        import json
        
        jams_dict = {
            'annotations': [],
            'file_metadata': {
                'duration': jam.file_metadata.duration,
                'identifiers': {},
                'jams_version': '0.3.4'
            },
            'sandbox': dict(jam.sandbox)
        }
        
        # Manually serialize each annotation
        for ann in jam.annotations:
            ann_dict = {
                'namespace': ann.namespace,
                'data': [],
                'annotation_metadata': {
                    'curator': {},
                    'annotator': {},
                    'version': '',
                    'corpus': '',
                    'annotation_tools': '',
                    'annotation_rules': '',
                    'validation': '',
                    'data_source': ''
                },
                'sandbox': {}
            }
            
            # Serialize observation data
            for obs in ann.data:
                obs_dict = {
                    'time': float(obs.time),
                    'duration': float(obs.duration),
                    'value': obs.value,
                    'confidence': float(obs.confidence) if obs.confidence is not None else None
                }
                ann_dict['data'].append(obs_dict)
            
            jams_dict['annotations'].append(ann_dict)
        
        # Write to file
        with open(jams_output_path, 'w') as f:
            json.dump(jams_dict, f, indent=2)
        
        import os
        if os.path.exists(jams_output_path):
            print(f"✓ JAMS saved to: {os.path.abspath(jams_output_path)}")
            print(f"  File size: {os.path.getsize(jams_output_path)} bytes")
            return jams_output_path
        else:
            print(f"✗ JAMS file was NOT created")
            return None
            
    except Exception as e:
        print(f"Warning: Could not save JAMS file: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with MusicXML generation...")