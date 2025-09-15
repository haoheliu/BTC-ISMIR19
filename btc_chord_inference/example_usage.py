#!/usr/bin/env python3
"""
Simple Usage Example for BTC Chord Recognition
==============================================

This script shows how to use the BTCChordRecognizer in your own code.
"""

import sys
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference import BTCChordRecognizer


def main():
    print("BTC Chord Recognition - Usage Example")
    print("=" * 40)
    
    # Initialize the recognizer with major/minor vocabulary
    print("1. Initializing recognizer...")
    recognizer = BTCChordRecognizer(vocabulary_type='major_minor')
    
    # Example: Process a single audio file
    audio_file = "../test/example.mp3"
    
    if Path(audio_file).exists():
        print(f"2. Processing audio file: {audio_file}")
        results = recognizer.recognize_chords(audio_file, save_results=False)
        
        print(f"   Song length: {results['song_length']:.2f} seconds")
        print(f"   Number of chord segments: {len(results['chord_segments'])}")
        
        # Show first 5 chord segments
        print("\n3. First 5 chord segments:")
        for i, segment in enumerate(results['chord_segments'][:5]):
            duration = segment['end_time'] - segment['start_time']
            print(f"   {segment['start_time']:6.2f}s - {segment['end_time']:6.2f}s ({duration:5.2f}s): {segment['chord']}")
        
        # Show chord statistics
        chord_counts = {}
        for segment in results['chord_segments']:
            chord = segment['chord']
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
        
        print("\n4. Most common chords:")
        for chord, count in sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {chord:15s}: {count:3d} segments")
        
        print("\n✅ Example completed successfully!")
        print("\nTo use in your own code:")
        print("   from inference import BTCChordRecognizer")
        print("   recognizer = BTCChordRecognizer()")
        print("   results = recognizer.recognize_chords('your_audio.mp3')")
        
    else:
        print(f"❌ Audio file not found: {audio_file}")
        print("Please provide a valid audio file path.")


if __name__ == "__main__":
    main()
