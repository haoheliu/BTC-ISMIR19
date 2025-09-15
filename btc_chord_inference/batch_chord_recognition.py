#!/usr/bin/env python3
"""
BTC Chord Recognition - Batch Processing Example
===============================================

This script demonstrates how to use the BTC chord recognition model for batch processing
of multiple audio waveforms. Perfect for processing audio collections or datasets.

Usage:
    python batch_chord_recognition.py --audio_files song1.mp3 song2.mp3 song3.mp3
    python batch_chord_recognition.py --audio_dir /path/to/audio/folder
"""

import sys
import argparse
import numpy as np
import librosa 
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference import BTCChordRecognizer


def load_audio_files(audio_paths):
    """
    Load multiple audio files as waveforms
    
    Args:
        audio_paths (list): List of paths to audio files
        
    Returns:
        tuple: (waveforms, sample_rates, audio_names)
    """
    waveforms = []
    sample_rates = []
    audio_names = []
    
    print(f"Loading {len(audio_paths)} audio files...")
    
    for i, audio_path in enumerate(audio_paths):
        if not Path(audio_path).exists():
            print(f"⚠️  Warning: Audio file not found: {audio_path}")
            continue
            
        print(f"   Loading {i+1}/{len(audio_paths)}: {Path(audio_path).name}")
        
        try:
            waveform, sr = librosa.load(audio_path, sr=None, mono=True)
            waveforms.append(waveform)
            sample_rates.append(sr)
            audio_names.append(Path(audio_path).stem)
            
            print(f"      {len(waveform)} samples at {sr}Hz ({len(waveform)/sr:.1f}s)")
            
        except Exception as e:
            print(f"      ❌ Error loading {audio_path}: {e}")
            continue
    
    print(f"✅ Successfully loaded {len(waveforms)} audio files")
    return waveforms, sample_rates, audio_names


def get_audio_files_from_dir(audio_dir, extensions=('.mp3', '.wav', '.m4a', '.flac')):
    """
    Get all audio files from a directory
    
    Args:
        audio_dir (str): Directory path
        extensions (tuple): Audio file extensions to include
        
    Returns:
        list: List of audio file paths
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"Directory not found: {audio_dir}")
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))
        audio_files.extend(audio_dir.glob(f"*{ext.upper()}"))
    
    return sorted([str(f) for f in audio_files])


def display_results(results):
    """
    Display batch processing results in a clean format
    """
    print("\n" + "="*60)
    print("BATCH CHORD RECOGNITION RESULTS")
    print("="*60)
    
    total_segments = 0
    all_chords = []
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['audio_file']}")
        print(f"   Duration: {result['song_length']:.1f}s")
        print(f"   Sample rate: {result['sample_rate']} Hz")
        print(f"   Chord segments: {len(result['chord_segments'])}")
        
        # Show chord progression
        if result['chord_segments']:
            print("   Chord progression:")
            for segment in result['chord_segments'][:5]:  # Show first 5
                duration = segment['end_time'] - segment['start_time']
                print(f"     {segment['start_time']:6.1f}s - {segment['end_time']:6.1f}s ({duration:4.1f}s): {segment['chord']}")
            
            if len(result['chord_segments']) > 5:
                print(f"     ... and {len(result['chord_segments']) - 5} more segments")
        
        total_segments += len(result['chord_segments'])
        all_chords.extend([seg['chord'] for seg in result['chord_segments']])
    
    # Show overall statistics
    print(f"\n" + "-"*40)
    print(f"OVERALL STATISTICS")
    print(f"-"*40)
    print(f"Total files processed: {len(results)}")
    print(f"Total chord segments: {total_segments}")
    
    if all_chords:
        chord_counts = {}
        for chord in all_chords:
            chord_counts[chord] = chord_counts.get(chord, 0) + 1
        
        print(f"\nMost common chords:")
        for chord, count in sorted(chord_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {chord:15s}: {count:3d} segments")


def main():
    parser = argparse.ArgumentParser(description='BTC Chord Recognition - Batch Processing')
    parser.add_argument('--audio_files', nargs='+', 
                       help='List of audio files to process')
    parser.add_argument('--audio_dir', type=str,
                       help='Directory containing audio files to process')
    parser.add_argument('--vocabulary', type=str, default='major_minor',
                       choices=['major_minor', 'large_vocabulary'],
                       help='Vocabulary type: major_minor (25 chords) or large_vocabulary (170 chords)')
    parser.add_argument('--output_dir', type=str, default='./chord_results',
                       help='Output directory for results')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.audio_files and not args.audio_dir:
        print("❌ Error: Please provide either --audio_files or --audio_dir")
        parser.print_help()
        return
    
    if args.audio_files and args.audio_dir:
        print("❌ Error: Please provide either --audio_files or --audio_dir, not both")
        return
    
    print("BTC Chord Recognition - Batch Processing")
    print("=" * 50)
    
    # Get audio files
    if args.audio_files:
        audio_paths = args.audio_files
    else:
        audio_paths = get_audio_files_from_dir(args.audio_dir)
        print(f"Found {len(audio_paths)} audio files in {args.audio_dir}")
    
    if not audio_paths:
        print("❌ No audio files found to process")
        return
    
    # Load audio files
    waveforms, sample_rates, audio_names = load_audio_files(audio_paths)
    
    if not waveforms:
        print("❌ No audio files could be loaded")
        return
    
    # Initialize recognizer
    print(f"\nInitializing BTC Chord Recognizer with {args.vocabulary} vocabulary...")
    recognizer = BTCChordRecognizer(vocabulary_type=args.vocabulary)
    
    # Process waveforms in batch
    print(f"\nProcessing {len(waveforms)} waveforms in batch...")
    results = recognizer.recognize_waveforms_batch(
        waveforms, 
        sample_rates, 
        audio_names,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    # Display results
    display_results(results)
    
    if not args.no_save:
        print(f"\n✅ Results saved to: {args.output_dir}")
        print(f"   • .lab files: Chord labels with timestamps")
        print(f"   • .midi files: Musical representation of chords")
    
    print(f"\n🎵 Batch processing completed successfully!")


if __name__ == "__main__":
    main()
