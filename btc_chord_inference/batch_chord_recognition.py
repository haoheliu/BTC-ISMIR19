#!/usr/bin/env python3

import sys
import numpy as np
import librosa
from pathlib import Path

# Add the package to Python path
sys.path.insert(0, str(Path(__file__).parent))

from inference import BTCChordRecognizer

def main():
    # Load the specific audio file
    audio_file = "/Users/haoheliu/Downloads/file_0fcb979c_010154.wav"
    waveform, sample_rate = librosa.load(audio_file, sr=None, mono=True)
    
    # Create batch by repeating the same waveform multiple times
    batch_size = 3
    waveforms = [waveform] * batch_size
    sample_rates = [sample_rate] * batch_size
    audio_names = None
    
    # Initialize recognizer and process batch
    recognizer = BTCChordRecognizer(vocabulary_type='major_minor')
    results = recognizer.recognize_waveforms_batch(
        waveforms, 
        sample_rates, 
        audio_names,
        save_results=False
    )
    
    return results

if __name__ == "__main__":
    results = main()
    print(results)