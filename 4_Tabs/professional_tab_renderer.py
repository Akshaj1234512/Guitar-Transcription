"""
Professional Guitar Tablature Renderer
Converts JAMS objects to publication-quality SVG tablature
"""

import svgwrite
from svgwrite import cm, mm
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math


@dataclass
class TabNote:
    """Represents a single note in tablature"""
    time: float
    string: int  # 1-6 (1 = high e, 6 = low E)
    fret: int
    duration: float = 0.25  # quarter note by default
    techniques: List[str] = None
    
    def __post_init__(self):
        if self.techniques is None:
            self.techniques = []


@dataclass
class RenderConfig:
    """Configuration for tablature rendering"""
    # Spacing
    string_spacing: float = 20  # pixels between strings
    measure_width: float = 240  # pixels per measure
    staff_margin_left: float = 60
    staff_margin_top: float = 80
    staff_margin_bottom: float = 60
    line_spacing: float = 120  # vertical space between systems
    
    # Styling
    staff_line_width: float = 1.5
    bar_line_width: float = 2
    double_bar_width: float = 3
    staff_color: str = '#000000'
    
    # Typography
    fret_font_size: int = 14
    fret_font_family: str = 'Arial, sans-serif'
    fret_font_weight: str = 'bold'
    technique_font_size: int = 10
    technique_font_family: str = 'Arial, sans-serif'
    annotation_font_size: int = 11
    header_font_size: int = 12
    
    # Rhythm notation
    stem_length: float = 30
    stem_width: float = 1.5
    flag_width: float = 8
    
    # Layout
    measures_per_line: int = 4


class MusicLayout:
    """Handles spacing and positioning logic for tablature"""
    
    def __init__(self, config: RenderConfig, tempo: int = 120, time_signature: Tuple[int, int] = (4, 4)):
        self.config = config
        self.tempo = tempo
        self.time_signature = time_signature
        self.beat_duration = 60.0 / tempo
        self.measure_duration = time_signature[0] * self.beat_duration
        
    def calculate_note_position(self, note_time: float, measure_start_time: float) -> float:
        """Calculate horizontal position of a note within a measure (0.0 to 1.0)"""
        time_in_measure = note_time - measure_start_time
        return time_in_measure / self.measure_duration
    
    def group_notes_by_measure(self, notes: List[TabNote]) -> List[List[TabNote]]:
        """Group notes by measure number"""
        if not notes:
            return []
        
        max_time = max(n.time + n.duration for n in notes)
        total_measures = math.ceil(max_time / self.measure_duration)
        
        measures = [[] for _ in range(total_measures)]
        for note in notes:
            measure_idx = int(note.time / self.measure_duration)
            if measure_idx < total_measures:
                measures[measure_idx].append(note)
        
        # Sort notes within each measure
        for measure in measures:
            measure.sort(key=lambda n: (n.time, n.string))
        
        return measures
    
    def group_measures_by_line(self, measures: List[List[TabNote]]) -> List[List[List[TabNote]]]:
        """Group measures into lines for display"""
        lines = []
        for i in range(0, len(measures), self.config.measures_per_line):
            lines.append(measures[i:i + self.config.measures_per_line])
        return lines


class SVGTabRenderer:
    """Renders tablature to SVG format"""
    
    STRING_NAMES = ['E', 'A', 'D', 'G', 'B', 'e']  # Low to high
    TECHNIQUE_SYMBOLS = {
        'hammer_on': 'H',
        'pull_off': 'P',
        'slide_up': '/',
        'slide_down': '\\',
        'bend': 'b',
        'release': 'r',
        'vibrato': '~',
        'palm_mute': 'PM',
        'harmonic': '<>',
        'tapping': 'T',
    }
    
    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()
        
    def create_drawing(self, width: float, height: float) -> svgwrite.Drawing:
        """Create a new SVG drawing"""
        return svgwrite.Drawing(size=(f'{width}px', f'{height}px'))
    
    def draw_staff_lines(self, dwg: svgwrite.Drawing, x: float, y: float, width: float):
        """Draw the 6 horizontal staff lines"""
        staff_group = dwg.g(id='staff-lines')
        
        for string_idx in range(6):
            line_y = y + (string_idx * self.config.string_spacing)
            staff_group.add(dwg.line(
                start=(x, line_y),
                end=(x + width, line_y),
                stroke=self.config.staff_color,
                stroke_width=self.config.staff_line_width
            ))
        
        return staff_group
    
    def draw_string_labels(self, dwg: svgwrite.Drawing, x: float, y: float):
        """Draw string names on the left side"""
        label_group = dwg.g(id='string-labels')
        
        for string_idx, string_name in enumerate(self.STRING_NAMES):
            label_y = y + (string_idx * self.config.string_spacing)
            label_group.add(dwg.text(
                string_name,
                insert=(x - 20, label_y + 5),
                font_size=self.config.fret_font_size,
                font_family=self.config.fret_font_family,
                font_weight='bold',
                text_anchor='middle',
                fill=self.config.staff_color
            ))
        
        return label_group
    
    def draw_bar_line(self, dwg: svgwrite.Drawing, x: float, y: float, 
                      is_double: bool = False, is_final: bool = False):
        """Draw a vertical bar line"""
        height = 5 * self.config.string_spacing
        
        if is_final:
            # Double bar line for ending
            bar_group = dwg.g(id='final-bar')
            line1 = dwg.line(
                start=(x - 3, y),
                end=(x - 3, y + height),
                stroke=self.config.staff_color,
                stroke_width=self.config.bar_line_width
            )
            line2 = dwg.line(
                start=(x, y),
                end=(x, y + height),
                stroke=self.config.staff_color,
                stroke_width=self.config.double_bar_width
            )
            bar_group.add(line1)
            bar_group.add(line2)
            return bar_group
        elif is_double:
            # Double bar line for sections
            bar_group = dwg.g(id='double-bar')
            line1 = dwg.line(
                start=(x - 2, y),
                end=(x - 2, y + height),
                stroke=self.config.staff_color,
                stroke_width=self.config.bar_line_width
            )
            line2 = dwg.line(
                start=(x, y),
                end=(x, y + height),
                stroke=self.config.staff_color,
                stroke_width=self.config.bar_line_width
            )
            bar_group.add(line1)
            bar_group.add(line2)
            return bar_group
        else:
            # Single bar line
            return dwg.line(
                start=(x, y),
                end=(x, y + height),
                stroke=self.config.staff_color,
                stroke_width=self.config.bar_line_width
            )
    
    def draw_note(self, dwg: svgwrite.Drawing, x: float, y: float, 
                  string_idx: int, fret: int, techniques: List[str]):
        """Draw a single note (fret number) on the staff"""
        note_group = dwg.g(id='note')
        
        # Calculate position on the string line
        note_y = y + (string_idx * self.config.string_spacing)
        
        # Draw fret number
        fret_text = str(fret)
        note_group.add(dwg.text(
            fret_text,
            insert=(x, note_y + 5),
            font_size=self.config.fret_font_size,
            font_family=self.config.fret_font_family,
            font_weight=self.config.fret_font_weight,
            text_anchor='middle',
            fill=self.config.staff_color
        ))
        
        # Draw technique symbols above
        if techniques:
            tech_y = y - 15
            # Filter out None values and get symbols, use original string if not found
            tech_symbols = []
            for t in techniques:
                if t:  # Skip None or empty strings
                    symbol = self.TECHNIQUE_SYMBOLS.get(t, t)
                    if symbol:  # Only add non-empty symbols
                        tech_symbols.append(symbol)
            
            if tech_symbols:  # Only draw if we have valid symbols
                tech_text = ' '.join(tech_symbols)
                note_group.add(dwg.text(
                    tech_text,
                    insert=(x, tech_y),
                    font_size=self.config.technique_font_size,
                    font_family=self.config.technique_font_family,
                    text_anchor='middle',
                    fill=self.config.staff_color,
                    font_style='italic'
                ))
        
        return note_group
    
    def draw_rhythm_stem(self, dwg: svgwrite.Drawing, x: float, y: float, duration: float):
        """Draw rhythm notation stem below the staff"""
        stem_group = dwg.g(id='rhythm-stem')
        
        # Start position below the staff
        stem_start_y = y + (5 * self.config.string_spacing) + 5
        stem_end_y = stem_start_y + self.config.stem_length
        
        # Draw stem
        stem_group.add(dwg.line(
            start=(x, stem_start_y),
            end=(x, stem_end_y),
            stroke=self.config.staff_color,
            stroke_width=self.config.stem_width
        ))
        
        # Add flags/beams based on duration
        if duration <= 0.125:  # Eighth note or smaller
            # Draw flag
            flag_path = f'M {x} {stem_end_y} q {self.config.flag_width} -8, {self.config.flag_width} -15'
            stem_group.add(dwg.path(
                d=flag_path,
                stroke=self.config.staff_color,
                stroke_width=self.config.stem_width,
                fill='none'
            ))
            
            if duration <= 0.0625:  # Sixteenth note
                flag_path2 = f'M {x} {stem_end_y - 5} q {self.config.flag_width} -8, {self.config.flag_width} -15'
                stem_group.add(dwg.path(
                    d=flag_path2,
                    stroke=self.config.staff_color,
                    stroke_width=self.config.stem_width,
                    fill='none'
                ))
        
        return stem_group
    
    def draw_measure(self, dwg: svgwrite.Drawing, x: float, y: float, 
                     notes: List[TabNote], measure_start_time: float, layout: MusicLayout):
        """Draw a complete measure with notes"""
        measure_group = dwg.g(id='measure')
        measure_width = self.config.measure_width
        
        for note in notes:
            # Calculate horizontal position within measure
            position_ratio = layout.calculate_note_position(note.time, measure_start_time)
            note_x = x + (position_ratio * measure_width)
            
            # Convert string number (1-6) to index (0-5)
            string_idx = 6 - note.string
            
            # Draw the note
            note_element = self.draw_note(dwg, note_x, y, string_idx, note.fret, note.techniques)
            measure_group.add(note_element)
            
            # Draw rhythm notation
            rhythm_element = self.draw_rhythm_stem(dwg, note_x, y, note.duration)
            measure_group.add(rhythm_element)
        
        return measure_group
    
    def draw_header(self, dwg: svgwrite.Drawing, title: str = "", 
                    tempo: int = 120, time_signature: Tuple[int, int] = (4, 4),
                    capo: Optional[int] = None):
        """Draw header with title and musical information"""
        header_group = dwg.g(id='header')
        
        y = 30
        
        # Title
        if title:
            header_group.add(dwg.text(
                title,
                insert=(self.config.staff_margin_left, y),
                font_size=16,
                font_family=self.config.fret_font_family,
                font_weight='bold',
                fill=self.config.staff_color
            ))
            y += 25
        
        # Tempo and time signature
        info_text = f"♩ = {tempo}     {time_signature[0]}/{time_signature[1]}"
        if capo:
            info_text += f"     Capo: fret {capo}"
        
        header_group.add(dwg.text(
            info_text,
            insert=(self.config.staff_margin_left, y),
            font_size=self.config.header_font_size,
            font_family=self.config.fret_font_family,
            fill=self.config.staff_color
        ))
        
        return header_group
    
    def draw_legend(self, dwg: svgwrite.Drawing, y: float):
        """Draw technique legend at the bottom"""
        legend_group = dwg.g(id='legend')
        
        legend_text = "H = hammer-on  •  P = pull-off  •  b = bend  •  / = slide up  •  \\ = slide down  •  ~ = vibrato"
        
        legend_group.add(dwg.text(
            legend_text,
            insert=(self.config.staff_margin_left, y),
            font_size=10,
            font_family=self.config.fret_font_family,
            font_style='italic',
            fill='#666666'
        ))
        
        return legend_group
    
    def render(self, notes: List[TabNote], output_path: str, 
               title: str = "", tempo: int = 120, 
               time_signature: Tuple[int, int] = (4, 4),
               capo: Optional[int] = None) -> str:
        """
        Render complete tablature to SVG file
        
        Args:
            notes: List of TabNote objects
            output_path: Path to save SVG file
            title: Optional title for the piece
            tempo: Tempo in BPM
            time_signature: Tuple of (beats_per_measure, beat_unit)
            capo: Optional capo position
            
        Returns:
            Path to the created SVG file
        """
        # Create layout manager
        layout = MusicLayout(self.config, tempo, time_signature)
        
        # Group notes by measure and line
        measures = layout.group_notes_by_measure(notes)
        lines = layout.group_measures_by_line(measures)
        
        # Calculate canvas size
        canvas_width = (self.config.staff_margin_left + 
                       (self.config.measures_per_line * self.config.measure_width) + 
                       60)
        
        canvas_height = (self.config.staff_margin_top + 
                        (len(lines) * (5 * self.config.string_spacing + self.config.line_spacing)) +
                        self.config.staff_margin_bottom)
        
        # Create drawing
        dwg = self.create_drawing(canvas_width, canvas_height)
        
        # Add header
        header = self.draw_header(dwg, title, tempo, time_signature, capo)
        dwg.add(header)
        
        # Draw each line of tablature
        current_y = self.config.staff_margin_top
        measure_idx = 0
        
        for line_idx, line_measures in enumerate(lines):
            line_group = dwg.g(id=f'line-{line_idx}')
            
            # Starting X position
            current_x = self.config.staff_margin_left
            
            # Draw string labels
            labels = self.draw_string_labels(dwg, current_x, current_y)
            line_group.add(labels)
            
            # Draw staff lines for this line
            staff_width = len(line_measures) * self.config.measure_width
            staff = self.draw_staff_lines(dwg, current_x, current_y, staff_width)
            line_group.add(staff)
            
            # Draw initial bar line
            bar = self.draw_bar_line(dwg, current_x, current_y)
            line_group.add(bar)
            
            # Draw each measure
            for m_idx, measure_notes in enumerate(line_measures):
                measure_start_time = measure_idx * layout.measure_duration
                
                measure = self.draw_measure(
                    dwg, current_x, current_y, 
                    measure_notes, measure_start_time, layout
                )
                line_group.add(measure)
                
                # Move to next measure
                current_x += self.config.measure_width
                
                # Draw bar line
                is_final = (measure_idx == len(measures) - 1)
                bar = self.draw_bar_line(dwg, current_x, current_y, is_final=is_final)
                line_group.add(bar)
                
                measure_idx += 1
            
            dwg.add(line_group)
            
            # Move to next line
            current_y += (5 * self.config.string_spacing) + self.config.line_spacing
        
        # Add legend
        legend_y = current_y - self.config.line_spacing + 40
        legend = self.draw_legend(dwg, legend_y)
        dwg.add(legend)
        
        # Save to file
        dwg.saveas(output_path)
        return output_path


def jam_to_svg(jam, output_path: str, title: str = "", 
               tempo: int = 120, time_signature: Tuple[int, int] = (4, 4),
               capo: Optional[int] = None,
               config: RenderConfig = None) -> str:
    """
    Convert a JAMS object into professional SVG tablature.
    
    Args:
        jam: JAMS object containing tab_note annotations
        output_path: Path to save the SVG file
        title: Optional title for the piece
        tempo: Tempo in BPM
        time_signature: Tuple of (beats_per_measure, beat_unit)
        capo: Optional capo position
        config: Optional custom rendering configuration
        
    Returns:
        Path to the created SVG file
    """
    # Extract notes from JAMS object
    notes = []
    for ann in jam.annotations:
        if ann.namespace == "tab_note":
            for obs in ann.data:
                val = obs.value
                duration = obs.duration if hasattr(obs, 'duration') else 0.25
                
                # Validate and clean techniques
                techniques = val.get('techniques', [])
                if techniques is None:
                    techniques = []
                elif not isinstance(techniques, list):
                    techniques = [techniques]
                
                # Filter out None values and empty strings
                techniques = [t for t in techniques if t]
                
                # Validate string number
                string_num = val.get('string', 1)
                if not (1 <= string_num <= 6):
                    print(f"Warning: Invalid string number {string_num}, using 1")
                    string_num = 1
                
                # Validate fret number
                fret_num = val.get('fret', 0)
                if fret_num < 0:
                    print(f"Warning: Negative fret number {fret_num}, using 0")
                    fret_num = 0
                
                notes.append(TabNote(
                    time=obs.time,
                    string=string_num,
                    fret=fret_num,
                    duration=duration,
                    techniques=techniques
                ))
    
    if not notes:
        raise ValueError("No tablature data found in JAMS object.")
    
    # Sort notes by time
    notes.sort(key=lambda x: x.time)
    
    # Create renderer and render
    renderer = SVGTabRenderer(config)
    return renderer.render(notes, output_path, title, tempo, time_signature, capo)