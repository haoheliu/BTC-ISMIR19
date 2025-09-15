#!/usr/bin/env python3
"""
Test script for BTC Chord Recognition Inference
==============================================

This script demonstrates how to use the BTCChordRecognizer class.
"""

import sys
import argparse
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference import BTCChordRecognizer


def main():
    parser = argparse.ArgumentParser(description='Test BTC Chord Recognition')
    parser.add_argument('--audio_file', type=str, required=True,
                       help='Path to audio file (mp3 or wav)')
    parser.add_argument('--vocabulary', type=str, default='major_minor',
                       choices=['major_minor', 'large_vocabulary'],
                       help='Vocabulary type to use')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: same as audio file)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    print(f"Initializing BTC Chord Recognizer with {args.vocabulary} vocabulary...")
    recognizer = BTCChordRecognizer(vocabulary_type=args.vocabulary)
    
    # Recognize chords
    print(f"Processing audio file: {args.audio_file}")
    results = recognizer.recognize_chords(
        args.audio_file, 
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    # Display results
    print("\n" + "="*60)
    print("CHORD RECOGNITION RESULTS")
    print("="*60)
    print(f"Audio file: {results['audio_file']}")
    print(f"Vocabulary type: {results['vocabulary_type']}")
    print(f"Song length: {results['song_length']:.2f} seconds")
    print(f"Number of chord segments: {len(results['chord_segments'])}")
    
    print("\nChord sequence:")
    print("-" * 40)
    for i, segment in enumerate(results['chord_segments'][:10]):  # Show first 10
        duration = segment['end_time'] - segment['start_time']
        print(f"{segment['start_time']:6.2f}s - {segment['end_time']:6.2f}s ({duration:5.2f}s): {segment['chord']}")
    
    if len(results['chord_segments']) > 10:
        print(f"... and {len(results['chord_segments']) - 10} more segments")
    
    # Show chord statistics
    chord_counts = {}
    for segment in results['chord_segments']:
        chord = segment['chord']
        chord_counts[chord] = chord_counts.get(chord, 0) + 1
    
    print(f"\nChord statistics:")
    print("-" * 40)
    for chord, count in sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{chord:15s}: {count:3d} segments")
    
    if not args.no_save:
        print(f"\nResults saved to: {Path(args.audio_file).parent if args.output_dir is None else args.output_dir}")


if __name__ == "__main__":
    main()
