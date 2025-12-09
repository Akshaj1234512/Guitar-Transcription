"""
Debug utility for diagnosing JAMS file issues with tablature rendering
"""

import jams


def inspect_jams(jam_file_or_object):
    """
    Inspect a JAMS file and print diagnostic information.
    Helps identify issues before rendering.
    
    Args:
        jam_file_or_object: Path to JAMS file or JAMS object
    """
    # Load if path provided
    if isinstance(jam_file_or_object, str):
        print(f"Loading JAMS file: {jam_file_or_object}")
        jam = jams.load(jam_file_or_object)
    else:
        jam = jam_file_or_object
    
    print("\n" + "=" * 60)
    print("JAMS FILE INSPECTION")
    print("=" * 60)
    
    # Find tab_note annotations
    tab_annotations = [ann for ann in jam.annotations if ann.namespace == "tab_note"]
    
    if not tab_annotations:
        print("\n❌ ERROR: No tab_note annotations found!")
        print(f"   Available namespaces: {[ann.namespace for ann in jam.annotations]}")
        return
    
    print(f"\n✓ Found {len(tab_annotations)} tab_note annotation(s)")
    
    for ann_idx, ann in enumerate(tab_annotations):
        print(f"\n--- Annotation {ann_idx + 1} ---")
        print(f"Number of notes: {len(ann.data)}")
        
        if len(ann.data) == 0:
            print("❌ No notes in this annotation!")
            continue
        
        # Sample first few notes
        print("\nFirst 5 notes:")
        for i, obs in enumerate(ann.data[:5]):
            print(f"\n  Note {i+1}:")
            print(f"    time: {obs.time}")
            print(f"    duration: {obs.duration if hasattr(obs, 'duration') else 'N/A'}")
            print(f"    value: {obs.value}")
            
            # Check for issues
            val = obs.value
            
            # Check string
            string_num = val.get('string', None)
            if string_num is None:
                print(f"    ⚠️  WARNING: No 'string' field!")
            elif not (1 <= string_num <= 6):
                print(f"    ⚠️  WARNING: Invalid string number {string_num} (should be 1-6)")
            
            # Check fret
            fret_num = val.get('fret', None)
            if fret_num is None:
                print(f"    ⚠️  WARNING: No 'fret' field!")
            elif fret_num < 0:
                print(f"    ⚠️  WARNING: Negative fret number {fret_num}")
            
            # Check techniques
            techniques = val.get('techniques', [])
            if techniques:
                print(f"    techniques: {techniques}")
                # Check for None values
                if None in techniques:
                    print(f"    ⚠️  WARNING: None value in techniques list!")
                # Check for unknown techniques
                known = {'hammer_on', 'pull_off', 'slide_up', 'slide_down', 
                        'bend', 'release', 'vibrato', 'palm_mute', 'harmonic', 'tapping'}
                unknown = set(techniques) - known
                if unknown:
                    print(f"    ℹ️  INFO: Unknown techniques (will use as-is): {unknown}")
        
        # Statistics
        print(f"\n--- Statistics ---")
        all_strings = [obs.value.get('string') for obs in ann.data]
        all_frets = [obs.value.get('fret') for obs in ann.data]
        all_times = [obs.time for obs in ann.data]
        
        print(f"Time range: {min(all_times):.2f} - {max(all_times):.2f} seconds")
        print(f"Duration: {max(all_times) - min(all_times):.2f} seconds")
        print(f"Strings used: {set(s for s in all_strings if s is not None)}")
        print(f"Fret range: {min(f for f in all_frets if f is not None)} - {max(f for f in all_frets if f is not None)}")
        
        # Check for common issues
        print(f"\n--- Issue Check ---")
        
        # None values
        none_strings = sum(1 for s in all_strings if s is None)
        none_frets = sum(1 for f in all_frets if f is None)
        if none_strings > 0:
            print(f"❌ {none_strings} notes missing 'string' field")
        if none_frets > 0:
            print(f"❌ {none_frets} notes missing 'fret' field")
        
        # Invalid ranges
        invalid_strings = sum(1 for s in all_strings if s is not None and not (1 <= s <= 6))
        invalid_frets = sum(1 for f in all_frets if f is not None and f < 0)
        if invalid_strings > 0:
            print(f"⚠️  {invalid_strings} notes with invalid string numbers")
        if invalid_frets > 0:
            print(f"⚠️  {invalid_frets} notes with negative fret numbers")
        
        # Technique issues
        all_techniques = []
        for obs in ann.data:
            techs = obs.value.get('techniques', [])
            if techs:
                all_techniques.extend(techs)
        
        none_techniques = sum(1 for t in all_techniques if t is None)
        if none_techniques > 0:
            print(f"⚠️  {none_techniques} None values in techniques")
        
        if none_strings == 0 and none_frets == 0 and invalid_strings == 0 and invalid_frets == 0 and none_techniques == 0:
            print("✓ No critical issues found!")
    
    print("\n" + "=" * 60)


def test_render_sample(jam_file_or_object, output_path='test_output.svg'):
    """
    Try to render the JAMS file and report any errors.
    
    Args:
        jam_file_or_object: Path to JAMS file or JAMS object
        output_path: Where to save the test output
    """
    from professional_tab_renderer import jam_to_svg
    
    # Load if path provided
    if isinstance(jam_file_or_object, str):
        print(f"Loading JAMS file: {jam_file_or_object}")
        jam = jams.load(jam_file_or_object)
    else:
        jam = jam_file_or_object
    
    print("\n" + "=" * 60)
    print("ATTEMPTING TO RENDER")
    print("=" * 60)
    
    try:
        result = jam_to_svg(
            jam=jam,
            output_path=output_path,
            title="Test Render",
            tempo=120
        )
        print(f"\n✓ SUCCESS! Created: {result}")
        print(f"  Open this file in a browser to view the tablature")
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR during rendering:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return None


def fix_jams_issues(jam):
    """
    Attempt to automatically fix common issues in a JAMS object.
    Returns a cleaned version.
    
    Args:
        jam: JAMS object to fix
        
    Returns:
        Fixed JAMS object
    """
    print("\n" + "=" * 60)
    print("ATTEMPTING TO FIX ISSUES")
    print("=" * 60)
    
    fixed_count = 0
    
    for ann in jam.annotations:
        if ann.namespace == "tab_note":
            for obs in ann.data:
                val = obs.value
                
                # Fix missing string
                if 'string' not in val or val['string'] is None:
                    val['string'] = 1
                    fixed_count += 1
                    print(f"Fixed: Added missing string=1 at time {obs.time}")
                
                # Fix invalid string
                elif not (1 <= val['string'] <= 6):
                    old = val['string']
                    val['string'] = max(1, min(6, abs(val['string'])))
                    fixed_count += 1
                    print(f"Fixed: Changed string from {old} to {val['string']} at time {obs.time}")
                
                # Fix missing fret
                if 'fret' not in val or val['fret'] is None:
                    val['fret'] = 0
                    fixed_count += 1
                    print(f"Fixed: Added missing fret=0 at time {obs.time}")
                
                # Fix negative fret
                elif val['fret'] < 0:
                    old = val['fret']
                    val['fret'] = 0
                    fixed_count += 1
                    print(f"Fixed: Changed fret from {old} to 0 at time {obs.time}")
                
                # Fix techniques
                if 'techniques' in val:
                    if val['techniques'] is None:
                        val['techniques'] = []
                        fixed_count += 1
                    elif not isinstance(val['techniques'], list):
                        val['techniques'] = [val['techniques']]
                        fixed_count += 1
                    else:
                        # Remove None values
                        old_len = len(val['techniques'])
                        val['techniques'] = [t for t in val['techniques'] if t is not None and t != '']
                        if len(val['techniques']) < old_len:
                            fixed_count += 1
                            print(f"Fixed: Removed None/empty from techniques at time {obs.time}")
    
    print(f"\n✓ Fixed {fixed_count} issue(s)")
    return jam


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python debug_jams.py <jams_file>")
        print("\nThis will:")
        print("  1. Inspect the JAMS file for issues")
        print("  2. Attempt to render it")
        print("  3. Offer to fix issues if found")
    else:
        jams_file = sys.argv[1]
        
        # Inspect
        inspect_jams(jams_file)
        
        # Try to render
        result = test_render_sample(jams_file, 'debug_output.svg')
        
        if result is None:
            print("\n" + "=" * 60)
            response = input("Would you like to try fixing the issues? (y/n): ")
            if response.lower() == 'y':
                jam = jams.load(jams_file)
                fixed_jam = fix_jams_issues(jam)
                
                # Try again
                print("\nAttempting to render with fixed data...")
                test_render_sample(fixed_jam, 'debug_output_fixed.svg')
