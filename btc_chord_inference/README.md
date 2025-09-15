# BTC Chord Recognition - Batch Processing Package

A clean, reusable package for batch chord recognition using the Bi-directional Transformer for Chord Recognition (BTC) model from ISMIR 2019.

## Features

- **Batch Processing**: Process multiple audio waveforms efficiently
- **Two vocabulary types**: Major/minor chords (25 classes) or large vocabulary (170 classes)
- **Multiple output formats**: Chord labels (.lab), MIDI files, and Python dictionaries
- **Real-time capable**: Waveform API enables real-time processing
- **GPU/CPU support**: Automatic device detection and model loading
- **Flexible audio handling**: Automatic resampling and mono conversion

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Copy the package**: The `btc_chord_inference` folder contains everything you need.

## Quick Start

### Batch Processing Multiple Audio Files

```bash
# Process specific audio files
python batch_chord_recognition.py --audio_files song1.mp3 song2.mp3 song3.mp3

# Process all audio files in a directory
python batch_chord_recognition.py --audio_dir /path/to/audio/folder

# Use large vocabulary mode
python batch_chord_recognition.py --audio_dir /path/to/audio --vocabulary large_vocabulary
```

### Python API Usage

```python
import librosa
from inference import BTCChordRecognizer

# Initialize recognizer
recognizer = BTCChordRecognizer(vocabulary_type='major_minor')

# Load multiple audio files
waveforms = []
sample_rates = []
audio_names = []

for audio_file in ['song1.mp3', 'song2.mp3', 'song3.mp3']:
    waveform, sr = librosa.load(audio_file, sr=None, mono=True)
    waveforms.append(waveform)
    sample_rates.append(sr)
    audio_names.append(Path(audio_file).stem)

# Process all waveforms in batch
results = recognizer.recognize_waveforms_batch(waveforms, sample_rates, audio_names)

# Access results
for result in results:
    print(f"Processed: {result['audio_file']}")
    print(f"Chords: {len(result['chord_segments'])} segments")
    for segment in result['chord_segments']:
        print(f"  {segment['start_time']:.1f}s - {segment['end_time']:.1f}s: {segment['chord']}")
```

## API Reference

### BTCChordRecognizer

#### `__init__(vocabulary_type='major_minor', config_path=None)`

Initialize the chord recognizer.

**Parameters:**
- `vocabulary_type` (str): Either `'major_minor'` (25 chord classes) or `'large_vocabulary'` (170 chord classes)
- `config_path` (str, optional): Path to configuration file. Uses default if None.

#### `recognize_waveforms_batch(waveforms, sample_rates, audio_names=None, save_results=True, output_dir=None)`

Recognize chords from multiple raw audio waveforms in batch.

**Parameters:**
- `waveforms` (list): List of raw audio waveforms (each is 1D numpy array)
- `sample_rates` (list): List of sample rates corresponding to each waveform
- `audio_names` (list, optional): List of names for each waveform
- `save_results` (bool): Whether to save results to files
- `output_dir` (str, optional): Output directory for results

**Returns:**
- `list`: List of result dictionaries, one for each waveform

**Benefits:**
- Process multiple waveforms efficiently
- Consistent API with single waveform processing
- Organized results structure
- Perfect for processing audio collections

## Output Formats

### Chord Labels (.lab)
```
0.000 10.000 N
10.000 11.944 G#
11.944 16.759 D#
16.759 17.407 N
...
```

### MIDI Files
MIDI files are automatically generated with chord notes mapped to piano pitches.

### Python Dictionary
```python
{
    'audio_file': 'song_name (waveform)',
    'vocabulary_type': 'major_minor',
    'song_length': 257.22,
    'sample_rate': 44100,
    'num_samples': 11342585,
    'chord_segments': [
        {
            'start_time': 0.0,
            'end_time': 10.0,
            'chord': 'N'
        },
        {
            'start_time': 10.0,
            'end_time': 11.944,
            'chord': 'G#'
        },
        ...
    ]
}
```

## Supported Audio Formats

- MP3, WAV, M4A, FLAC
- Other formats supported by librosa

## Vocabulary Types

### Major/Minor Vocabulary (25 classes)
- Major chords: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
- Minor chords: C:min, C#:min, D:min, D#:min, E:min, F:min, F#:min, G:min, G#:min, A:min, A#:min, B:min
- No chord: N

### Large Vocabulary (170 classes)
Extended chord vocabulary including:
- Major, minor, diminished, augmented triads
- Seventh chords (major7, minor7, dominant7, etc.)
- Extended chords (9th, 11th, 13th)
- Suspended chords
- No chord: N

## Examples

### Process Audio Directory
```bash
python batch_chord_recognition.py --audio_dir /path/to/music --vocabulary large_vocabulary --output_dir results/
```

### Process Specific Files
```bash
python batch_chord_recognition.py --audio_files song1.mp3 song2.mp3 --vocabulary major_minor
```

### Python Integration
```python
from inference import BTCChordRecognizer
import librosa

recognizer = BTCChordRecognizer(vocabulary_type='large_vocabulary')
waveforms, sample_rates, names = load_audio_collection()
results = recognizer.recognize_waveforms_batch(waveforms, sample_rates, names)
```

## Model Information

- **Architecture**: Bi-directional Transformer
- **Input**: Constant-Q Transform features (144 bins)
- **Training**: Trained on Isophonics, Robbie Williams, and UsPop2002 datasets
- **Performance**: State-of-the-art chord recognition accuracy

## Requirements

- Python 3.7+
- PyTorch 1.0+
- librosa
- mir_eval
- pretty_midi
- numpy
- pandas
- pyyaml

## License

This package contains code from the BTC-ISMIR19 repository. Please refer to the original LICENSE file for licensing terms.

## Citation

If you use this code, please cite the original paper:

```
@inproceedings{choi2019bi,
  title={A Bi-Directional Transformer for Musical Chord Recognition},
  author={Choi, Keunwoo and Fazekas, George and Sandler, Mark},
  booktitle={ISMIR},
  year={2019}
}
```