#!/usr/bin/env python3
"""
Test script for BTC Chord Recognition Waveform API
=================================================

This script demonstrates how to use the BTCChordRecognizer with raw audio waveforms.
"""

import sys
import argparse
import numpy as np
import librosa
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference import BTCChordRecognizer


def load_audio_as_waveform(audio_file, target_sr=None):
    """
    Load audio file as raw waveform
    
    Args:
        audio_file (str): Path to audio file
        target_sr (int, optional): Target sample rate. If None, uses original sample rate.
        
    Returns:
        tuple: (waveform, sample_rate)
    """
    waveform, sr = librosa.load(audio_file, sr=target_sr, mono=True)
    return waveform, sr


def generate_test_waveform(duration=10, sample_rate=22050, frequency=440):
    """
    Generate a simple test waveform (sine wave)
    
    Args:
        duration (float): Duration in seconds
        sample_rate (int): Sample rate
        frequency (float): Frequency of sine wave
        
    Returns:
        tuple: (waveform, sample_rate)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    waveform = np.sin(2 * np.pi * frequency * t)
    return waveform, sample_rate


def main():
    parser = argparse.ArgumentParser(description='Test BTC Chord Recognition Waveform API')
    parser.add_argument('--audio_file', type=str, default=None,
                       help='Path to audio file to load as waveform')
    parser.add_argument('--vocabulary', type=str, default='major_minor',
                       choices=['major_minor', 'large_vocabulary'],
                       help='Vocabulary type to use')
    parser.add_argument('--output_dir', type=str, default='./waveform_results',
                       help='Output directory for results')
    parser.add_argument('--generate_test', action='store_true',
                       help='Generate a test sine wave instead of loading audio file')
    parser.add_argument('--test_duration', type=float, default=10.0,
                       help='Duration of test waveform in seconds')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Initialize recognizer
    print(f"Initializing BTC Chord Recognizer with {args.vocabulary} vocabulary...")
    recognizer = BTCChordRecognizer(vocabulary_type=args.vocabulary)
    
    # Load or generate waveform
    if args.generate_test:
        print(f"Generating test sine wave ({args.test_duration}s)...")
        waveform, sample_rate = generate_test_waveform(
            duration=args.test_duration, 
            sample_rate=22050, 
            frequency=440
        )
        audio_name = f"test_sine_{args.test_duration}s"
    elif args.audio_file:
        if not Path(args.audio_file).exists():
            print(f"❌ Audio file not found: {args.audio_file}")
            return
        
        print(f"Loading audio file as waveform: {args.audio_file}")
        waveform, sample_rate = load_audio_as_waveform(args.audio_file)
        audio_name = Path(args.audio_file).stem
    else:
        print("❌ Please provide --audio_file or use --generate_test")
        return
    
    print(f"Waveform info: {len(waveform)} samples at {sample_rate}Hz")
    print(f"Duration: {len(waveform) / sample_rate:.2f} seconds")
    
    # Recognize chords from waveform
    print("Processing waveform...")
    results = recognizer.recognize_waveform(
        waveform, 
        sample_rate,
        save_results=not args.no_save,
        output_dir=args.output_dir,
        audio_name=audio_name
    )
    
    # Display results
    print("\n" + "="*60)
    print("WAVEFORM CHORD RECOGNITION RESULTS")
    print("="*60)
    print(f"Audio: {results['audio_file']}")
    print(f"Vocabulary type: {results['vocabulary_type']}")
    print(f"Sample rate: {results['sample_rate']} Hz")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Song length: {results['song_length']:.2f} seconds")
    print(f"Number of chord segments: {len(results['chord_segments'])}")
    
    if results['chord_segments']:
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
    else:
        print("\nNo chord segments detected.")
    
    if not args.no_save:
        print(f"\nResults saved to: {args.output_dir}")
    
    print("\n✅ Waveform API test completed successfully!")


if __name__ == "__main__":
    main()
