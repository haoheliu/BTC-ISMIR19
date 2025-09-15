#!/usr/bin/env python3
"""
Batch Waveform Processing Usage Example
======================================

This script demonstrates how to use the batch processing API for multiple waveforms.
"""

import sys
import numpy as np
import librosa
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference import BTCChordRecognizer


def main():
    print("BTC Chord Recognition - Batch Processing Example")
    print("=" * 50)
    
    # Initialize recognizer
    print("1. Initializing recognizer...")
    recognizer = BTCChordRecognizer(vocabulary_type='major_minor')
    
    # Generate multiple test waveforms
    print("\n2. Generating test waveforms...")
    waveforms = []
    sample_rates = []
    audio_names = []
    
    # Generate different chord progressions
    chord_progressions = [
        # C major progression
        [(261.63, 329.63, 392.00), (392.00, 493.88, 587.33), (220.00, 261.63, 329.63)],
        # G major progression
        [(392.00, 493.88, 587.33), (293.66, 369.99, 440.00), (220.00, 261.63, 329.63)],
        # A minor progression
        [(220.00, 261.63, 329.63), (293.66, 369.99, 440.00), (174.61, 220.00, 261.63)]
    ]
    
    chord_names = ['C_major_progression', 'G_major_progression', 'A_minor_progression']
    sample_rate = 22050
    duration_per_chord = 2.0
    
    for i, (progression, name) in enumerate(zip(chord_progressions, chord_names)):
        print(f"   Generating {name}...")
        waveform = np.array([])
        
        for freqs in progression:
            t = np.linspace(0, duration_per_chord, int(sample_rate * duration_per_chord), False)
            chord_waveform = np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)
            waveform = np.concatenate([waveform, chord_waveform])
        
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
        audio_names.append(name)
    
    print(f"   Generated {len(waveforms)} waveforms")
    
    # Process all waveforms in batch
    print("\n3. Processing waveforms in batch...")
    results = recognizer.recognize_waveforms_batch(
        waveforms, 
        sample_rates, 
        audio_names,
        save_results=True,
        output_dir='./batch_example_results'
    )
    
    # Display results
    print("\n4. Batch processing results:")
    print("-" * 40)
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['audio_file']}")
        print(f"   Duration: {result['song_length']:.1f}s")
        print(f"   Sample rate: {result['sample_rate']} Hz")
        print(f"   Chord segments: {len(result['chord_segments'])}")
        
        # Show chord progression
        print("   Chord progression:")
        for segment in result['chord_segments']:
            duration = segment['end_time'] - segment['start_time']
            print(f"     {segment['start_time']:5.1f}s - {segment['end_time']:5.1f}s ({duration:4.1f}s): {segment['chord']}")
    
    # Show chord statistics across all waveforms
    print("\n5. Overall chord statistics:")
    print("-" * 40)
    all_chords = []
    for result in results:
        for segment in result['chord_segments']:
            all_chords.append(segment['chord'])
    
    chord_counts = {}
    for chord in all_chords:
        chord_counts[chord] = chord_counts.get(chord, 0) + 1
    
    for chord, count in sorted(chord_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {chord:15s}: {count:3d} segments")
    
    print(f"\n✅ Batch processing example completed!")
    print(f"\nBatch API benefits:")
    print(f"   • Process multiple waveforms efficiently")
    print(f"   • Consistent API with single waveform processing")
    print(f"   • Organized results structure")
    print(f"   • Automatic file saving for all waveforms")
    print(f"   • Perfect for processing audio collections")


if __name__ == "__main__":
    main()
