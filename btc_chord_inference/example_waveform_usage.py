#!/usr/bin/env python3
"""
Waveform API Usage Example for BTC Chord Recognition
==================================================

This script shows how to use the BTCChordRecognizer with raw audio waveforms.
"""

import sys
import numpy as np
import librosa
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference import BTCChordRecognizer


def main():
    print("BTC Chord Recognition - Waveform API Usage Example")
    print("=" * 50)
    
    # Initialize the recognizer
    print("1. Initializing recognizer...")
    recognizer = BTCChordRecognizer(vocabulary_type='major_minor')
    
    # Example 1: Load audio file as waveform
    audio_file = "../test/example.mp3"
    
    if Path(audio_file).exists():
        print(f"\n2. Loading audio file as waveform: {audio_file}")
        waveform, sample_rate = librosa.load(audio_file, sr=None, mono=True)
        
        print(f"   Waveform shape: {waveform.shape}")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Duration: {len(waveform) / sample_rate:.2f} seconds")
        
        # Recognize chords from waveform
        print("\n3. Recognizing chords from waveform...")
        results = recognizer.recognize_waveform(
            waveform, 
            sample_rate, 
            save_results=False
        )
        
        print(f"   Number of chord segments: {len(results['chord_segments'])}")
        
        # Show first 5 chord segments
        print("\n4. First 5 chord segments:")
        for i, segment in enumerate(results['chord_segments'][:5]):
            duration = segment['end_time'] - segment['start_time']
            print(f"   {segment['start_time']:6.2f}s - {segment['end_time']:6.2f}s ({duration:5.2f}s): {segment['chord']}")
        
        # Show chord statistics
        chord_counts = {}
        for segment in results['chord_segments']:
            chord = segment['chord']
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
        
        print("\n5. Most common chords:")
        for chord, count in sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {chord:15s}: {count:3d} segments")
    
    # Example 2: Generate and process a synthetic waveform
    print("\n" + "="*50)
    print("6. Example with synthetic waveform:")
    
    # Generate a simple chord progression (C major, G major, A minor, F major)
    sample_rate = 22050
    duration_per_chord = 2.0  # 2 seconds per chord
    frequencies = [
        [261.63, 329.63, 392.00],  # C major (C, E, G)
        [392.00, 493.88, 587.33],  # G major (G, B, D)
        [220.00, 261.63, 329.63],  # A minor (A, C, E)
        [174.61, 220.00, 261.63]   # F major (F, A, C)
    ]
    
    chord_names = ['C', 'G', 'A:min', 'F']
    waveform = np.array([])
    
    for i, (freqs, chord_name) in enumerate(zip(frequencies, chord_names)):
        print(f"   Generating {chord_name} chord...")
        t = np.linspace(0, duration_per_chord, int(sample_rate * duration_per_chord), False)
        chord_waveform = np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)
        waveform = np.concatenate([waveform, chord_waveform])
    
    print(f"   Generated waveform: {len(waveform)} samples, {len(waveform)/sample_rate:.1f}s")
    
    # Recognize chords from synthetic waveform
    results_synth = recognizer.recognize_waveform(
        waveform, 
        sample_rate, 
        save_results=False,
        audio_name="synthetic_chord_progression"
    )
    
    print(f"   Detected {len(results_synth['chord_segments'])} chord segments:")
    for segment in results_synth['chord_segments']:
        duration = segment['end_time'] - segment['start_time']
        print(f"   {segment['start_time']:6.2f}s - {segment['end_time']:6.2f}s ({duration:5.2f}s): {segment['chord']}")
    
    print("\n✅ Waveform API examples completed successfully!")
    print("\nKey benefits of waveform API:")
    print("   • Process audio from any source (microphone, streaming, etc.)")
    print("   • No need to save audio to files")
    print("   • Real-time processing capabilities")
    print("   • Works with any sample rate (auto-resampling)")
    print("   • Memory-efficient for large audio data")


if __name__ == "__main__":
    main()
