"""
Utility to convert MusicXML to PDF automatically
"""

import subprocess
import os
import shutil


def find_musescore():
    """Find MuseScore executable on your system"""
    
    # Common paths
    possible_paths = [
        # macOS
        '/Applications/MuseScore 3.app/Contents/MacOS/mscore',
        '/Applications/MuseScore 4.app/Contents/MacOS/mscore',
        
        # Linux
        '/usr/bin/musescore',
        '/usr/bin/musescore3',
        '/usr/bin/mscore',
        
        # Windows
        r'C:\Program Files\MuseScore 3\bin\MuseScore3.exe',
        r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe',
    ]
    
    # Check if musescore is in PATH
    if shutil.which('musescore'):
        return 'musescore'
    if shutil.which('mscore'):
        return 'mscore'
    
    # Check common paths
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def musicxml_to_pdf(xml_file, pdf_file=None, musescore_path=None):
    """
    Convert MusicXML to PDF using MuseScore
    
    Args:
        xml_file: Path to MusicXML file
        pdf_file: Path to output PDF (default: same name as xml_file)
        musescore_path: Path to MuseScore executable (auto-detected if None)
    
    Returns:
        Path to created PDF file
    """
    
    # Auto-detect MuseScore if not provided
    if musescore_path is None:
        musescore_path = find_musescore()
    
    if musescore_path is None:
        raise FileNotFoundError(
            "MuseScore not found. Please install it from https://musescore.org/download\n"
            "Or specify the path manually: musicxml_to_pdf(xml, pdf, musescore_path='/path/to/musescore')"
        )
    
    # Default output path
    if pdf_file is None:
        pdf_file = xml_file.replace('.xml', '.pdf')
    
    # Convert using MuseScore
    try:
        # MuseScore 3 and 4 syntax
        subprocess.run([
            musescore_path,
            xml_file,
            '-o', pdf_file
        ], check=True, capture_output=True)
        
        print(f"✓ Created {pdf_file}")
        return pdf_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error converting: {e}")
        print(f"stderr: {e.stderr.decode()}")
        raise


def musicxml_to_png(xml_file, png_file=None, musescore_path=None, dpi=300):
    """
    Convert MusicXML to PNG using MuseScore
    
    Args:
        xml_file: Path to MusicXML file
        png_file: Path to output PNG
        musescore_path: Path to MuseScore executable
        dpi: Resolution (default 300 for high quality)
    
    Returns:
        Path to created PNG file
    """
    
    if musescore_path is None:
        musescore_path = find_musescore()
    
    if musescore_path is None:
        raise FileNotFoundError("MuseScore not found")
    
    if png_file is None:
        png_file = xml_file.replace('.xml', '.png')
    
    # Convert
    subprocess.run([
        musescore_path,
        xml_file,
        '-o', png_file,
        '-r', str(dpi)  # resolution
    ], check=True, capture_output=True)
    
    print(f"✓ Created {png_file}")
    return png_file


def view_musicxml(xml_file, musescore_path=None):
    """
    Open MusicXML file in MuseScore for viewing
    
    Args:
        xml_file: Path to MusicXML file
        musescore_path: Path to MuseScore executable
    """
    
    if musescore_path is None:
        musescore_path = find_musescore()
    
    if musescore_path is None:
        print("MuseScore not found.")
        print("You can view the file at: https://www.soundslice.com/musicxml-viewer/")
        return
    
    # Open in MuseScore
    subprocess.Popen([musescore_path, xml_file])
    print(f"Opening {xml_file} in MuseScore...")


def convert_all_xml_in_folder(folder='.', output_format='pdf'):
    """
    Convert all MusicXML files in a folder to PDF or PNG
    
    Args:
        folder: Folder containing .xml files
        output_format: 'pdf' or 'png'
    """
    
    musescore = find_musescore()
    if musescore is None:
        raise FileNotFoundError("MuseScore not found")
    
    xml_files = [f for f in os.listdir(folder) if f.endswith('.xml')]
    
    if not xml_files:
        print(f"No .xml files found in {folder}")
        return
    
    print(f"Converting {len(xml_files)} files to {output_format.upper()}...")
    
    for xml_file in xml_files:
        xml_path = os.path.join(folder, xml_file)
        output_path = xml_path.replace('.xml', f'.{output_format}')
        
        try:
            if output_format == 'pdf':
                musicxml_to_pdf(xml_path, output_path, musescore)
            elif output_format == 'png':
                musicxml_to_png(xml_path, output_path, musescore)
            print(f"  ✓ {xml_file} → {output_format.upper()}")
        except Exception as e:
            print(f"  ✗ {xml_file}: {e}")
    
    print(f"\n✓ Done! Converted {len(xml_files)} files")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("MusicXML Viewer/Converter Utility")
    print("=" * 70)
    
    # Check if MuseScore is installed
    musescore = find_musescore()
    if musescore:
        print(f"\n✓ MuseScore found: {musescore}")
    else:
        print("\n✗ MuseScore not found")
        print("  Download from: https://musescore.org/download")
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES:")
    print("=" * 70)
    
    print("\n# View a MusicXML file:")
    print("view_musicxml('output.xml')")
    
    print("\n# Convert to PDF:")
    print("musicxml_to_pdf('output.xml', 'beautiful_tabs.pdf')")
    
    print("\n# Convert to PNG (high resolution):")
    print("musicxml_to_png('output.xml', 'tabs.png', dpi=300)")
    
    print("\n# Convert all XML files in current folder:")
    print("convert_all_xml_in_folder('.', output_format='pdf')")
    
    print("\n# Or use from command line:")
    print("python musicxml_converter.py output.xml")
    
    # Command line usage
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
        if os.path.exists(xml_file):
            print(f"\nConverting {xml_file}...")
            pdf_file = musicxml_to_pdf(xml_file)
            print(f"✓ Created {pdf_file}")
        else:
            print(f"Error: File not found: {xml_file}")
