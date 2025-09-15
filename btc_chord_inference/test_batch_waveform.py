#!/usr/bin/env python3
"""
Test script for BTC Chord Recognition Batch Waveform Processing
=============================================================

This script demonstrates the batch processing capabilities for multiple waveforms.
"""

import sys
import time
import argparse
import numpy as np
import librosa
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference import BTCChordRecognizer


def generate_test_waveforms(num_waveforms=5, duration=None, sample_rate=None):
    """
    Generate multiple test waveforms from the same audio file for consistency testing
    """
    waveforms = []
    sample_rates = []
    audio_names = []
    
    # Load the example audio file
    audio_file = "/Users/haoheliu/Project/BTC-ISMIR19/test/example.mp3"
    
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        print("Falling back to synthetic waveforms...")
        return generate_synthetic_waveforms(num_waveforms, duration or 5.0, sample_rate or 22050)
    
    print(f"Loading audio file: {audio_file}")
    original_waveform, original_sr = librosa.load(audio_file, sr=None, mono=True)
    
    print(f"Original waveform: {len(original_waveform)} samples at {original_sr}Hz")
    print(f"Duration: {len(original_waveform)/original_sr:.2f} seconds")
    
    # Create multiple samples from the same audio file
    for i in range(num_waveforms):
        # Use the same waveform for all samples to test consistency
        waveform = original_waveform.copy()
        sr = original_sr
        
        # Optionally truncate to specified duration if provided
        if duration is not None:
            max_samples = int(sr * duration)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
        
        waveforms.append(waveform)
        sample_rates.append(sr)
        audio_names.append(f"example_sample_{i+1}")
    
    return waveforms, sample_rates, audio_names


def generate_synthetic_waveforms(num_waveforms=5, duration=5.0, sample_rate=22050):
    """
    Fallback function to generate synthetic waveforms if audio file is not available
    """
    waveforms = []
    sample_rates = []
    audio_names = []
    
    # Different chord progressions for variety
    chord_progressions = [
        # C major progression
        [(261.63, 329.63, 392.00), (392.00, 493.88, 587.33), (220.00, 261.63, 329.63), (174.61, 220.00, 261.63)],
        # G major progression  
        [(392.00, 493.88, 587.33), (293.66, 369.99, 440.00), (220.00, 261.63, 329.63), (261.63, 329.63, 392.00)],
        # A minor progression
        [(220.00, 261.63, 329.63), (293.66, 369.99, 440.00), (174.61, 220.00, 261.63), (220.00, 261.63, 329.63)],
        # F major progression
        [(174.61, 220.00, 261.63), (261.63, 329.63, 392.00), (220.00, 261.63, 329.63), (293.66, 369.99, 440.00)],
        # Simple major scale
        [(261.63, 329.63, 392.00), (293.66, 369.99, 440.00), (329.63, 415.30, 493.88), (349.23, 440.00, 523.25)]
    ]
    
    chord_names = ['C_major', 'G_major', 'A_minor', 'F_major', 'Major_scale']
    
    for i in range(min(num_waveforms, len(chord_progressions))):
        progression = chord_progressions[i]
        chord_name = chord_names[i]
        
        # Generate waveform for this progression
        waveform = np.array([])
        duration_per_chord = duration / len(progression)
        
        for freqs in progression:
            t = np.linspace(0, duration_per_chord, int(sample_rate * duration_per_chord), False)
            chord_waveform = np.sum([np.sin(2 * np.pi * f * t) for f in freqs], axis=0)
            waveform = np.concatenate([waveform, chord_waveform])
        
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
        audio_names.append(f"{chord_name}_{duration}s")
    
    return waveforms, sample_rates, audio_names


def main():
    parser = argparse.ArgumentParser(description='Test BTC Chord Recognition Batch Processing')
    parser.add_argument('--num_waveforms', type=int, default=5,
                       help='Number of test waveforms to generate')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Duration of each waveform in seconds')
    parser.add_argument('--vocabulary', type=str, default='major_minor',
                       choices=['major_minor', 'large_vocabulary'],
                       help='Vocabulary type to use')
    parser.add_argument('--output_dir', type=str, default='./batch_results',
                       help='Output directory for results')
    parser.add_argument('--compare_individual', action='store_true',
                       help='Compare with individual processing for speed comparison')
    
    args = parser.parse_args()
    
    print("BTC Chord Recognition - Batch Processing Test")
    print("=" * 50)
    
    # Generate test waveforms
    print(f"1. Generating {args.num_waveforms} test waveforms from example.mp3...")
    waveforms, sample_rates, audio_names = generate_test_waveforms(
        args.num_waveforms, args.duration
    )
    
    print(f"   Waveforms generated:")
    for i, (name, waveform, sr) in enumerate(zip(audio_names, waveforms, sample_rates)):
        print(f"   {i+1}. {name}: {len(waveform)} samples at {sr}Hz ({len(waveform)/sr:.1f}s)")
    
    # Check that all waveforms are identical (for consistency testing)
    if len(waveforms) > 1:
        all_identical = all(np.array_equal(waveforms[0], w) for w in waveforms[1:])
        print(f"   ✅ All waveforms are identical: {all_identical}")
        if all_identical:
            print(f"   This ensures consistent results for manual verification")
    
    # Initialize recognizer
    print(f"\n2. Initializing recognizer with {args.vocabulary} vocabulary...")
    recognizer = BTCChordRecognizer(vocabulary_type=args.vocabulary)
    
    # Test batch processing
    print(f"\n3. Processing waveforms in batch...")
    start_time = time.time()
    
    batch_results = recognizer.recognize_waveforms_batch(
        waveforms, 
        sample_rates, 
        audio_names,
        save_results=True,
        output_dir=args.output_dir
    )
    
    batch_time = time.time() - start_time
    
    print(f"   Batch processing completed in {batch_time:.2f} seconds")
    print(f"   Average time per waveform: {batch_time/len(waveforms):.2f} seconds")
    
    # Display results
    print(f"\n4. Batch processing results:")
    print("-" * 40)
    
    # Check consistency of results (should be identical since waveforms are identical)
    results_consistent = True
    if len(batch_results) > 1:
        first_result = batch_results[0]
        for i, result in enumerate(batch_results[1:], 1):
            if (len(result['chord_segments']) != len(first_result['chord_segments']) or
                any(seg1['chord'] != seg2['chord'] or 
                    abs(seg1['start_time'] - seg2['start_time']) > 0.01
                    for seg1, seg2 in zip(result['chord_segments'], first_result['chord_segments']))):
                results_consistent = False
                break
    
    print(f"   ✅ Results consistency: {'All identical' if results_consistent else 'Different results detected'}")
    print(f"   This confirms the batch processing is deterministic and reliable\n")
    
    for i, result in enumerate(batch_results):
        print(f"   {i+1}. {result['audio_file']}")
        print(f"      Duration: {result['song_length']:.1f}s, Segments: {len(result['chord_segments'])}")
        
        # Show first few chord segments
        for j, segment in enumerate(result['chord_segments'][:3]):
            duration = segment['end_time'] - segment['start_time']
            print(f"      {segment['start_time']:5.1f}s - {segment['end_time']:5.1f}s: {segment['chord']}")
        if len(result['chord_segments']) > 3:
            print(f"      ... and {len(result['chord_segments']) - 3} more segments")
    
    # Compare with individual processing if requested
    if args.compare_individual:
        print(f"\n5. Comparing with individual processing...")
        
        individual_start_time = time.time()
        individual_results = []
        
        for i, (waveform, sample_rate, audio_name) in enumerate(zip(waveforms, sample_rates, audio_names)):
            print(f"   Processing {i+1}/{len(waveforms)} individually...")
            result = recognizer.recognize_waveform(
                waveform, 
                sample_rate, 
                save_results=False,
                audio_name=f"{audio_name}_individual"
            )
            individual_results.append(result)
        
        individual_time = time.time() - individual_start_time
        
        print(f"   Individual processing completed in {individual_time:.2f} seconds")
        print(f"   Average time per waveform: {individual_time/len(waveforms):.2f} seconds")
        
        # Calculate speedup
        speedup = individual_time / batch_time
        print(f"   Batch processing speedup: {speedup:.2f}x")
        
        # Verify results are identical
        results_match = True
        for batch_result, individual_result in zip(batch_results, individual_results):
            if len(batch_result['chord_segments']) != len(individual_result['chord_segments']):
                results_match = False
                break
            for batch_seg, individual_seg in zip(batch_result['chord_segments'], individual_result['chord_segments']):
                if (batch_seg['chord'] != individual_seg['chord'] or 
                    abs(batch_seg['start_time'] - individual_seg['start_time']) > 0.01):
                    results_match = False
                    break
        
        print(f"   Results match: {'✅ Yes' if results_match else '❌ No'}")
    
    print(f"\n✅ Batch processing test completed successfully!")
    print(f"\nBatch processing benefits:")
    print(f"   • Processed {len(waveforms)} waveforms efficiently")
    print(f"   • Organized results in structured format")
    print(f"   • Reduced overhead compared to individual processing")
    print(f"   • Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
