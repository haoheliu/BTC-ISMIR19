# BTC Chord Recognition Inference Package

A clean, reusable package for chord recognition using the Bi-directional Transformer for Chord Recognition (BTC) model from ISMIR 2019.

## Features

- **Easy-to-use API**: Simple Python interface for chord recognition
- **Dual input modes**: Process audio files or raw waveforms directly
- **Two vocabulary types**: Major/minor chords (25 classes) or large vocabulary (170 classes)
- **Multiple output formats**: Chord labels (.lab), MIDI files, and Python dictionaries
- **Batch processing**: Process multiple audio files at once
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

### Basic Usage

**File-based processing:**
```python
from btc_chord_inference import BTCChordRecognizer

# Initialize recognizer
recognizer = BTCChordRecognizer(vocabulary_type='major_minor')

# Recognize chords from audio file
results = recognizer.recognize_chords('path/to/audio.mp3')

# Access results
print(f"Song length: {results['song_length']} seconds")
for segment in results['chord_segments']:
    print(f"{segment['start_time']:.2f}s - {segment['end_time']:.2f}s: {segment['chord']}")
```

**Waveform-based processing:**
```python
import librosa
from btc_chord_inference import BTCChordRecognizer

# Initialize recognizer
recognizer = BTCChordRecognizer(vocabulary_type='major_minor')

# Load audio as waveform
waveform, sample_rate = librosa.load('path/to/audio.mp3', sr=None, mono=True)

# Recognize chords from raw waveform
results = recognizer.recognize_waveform(waveform, sample_rate)

# Access results
print(f"Sample rate: {results['sample_rate']} Hz")
print(f"Number of samples: {results['num_samples']}")
for segment in results['chord_segments']:
    print(f"{segment['start_time']:.2f}s - {segment['end_time']:.2f}s: {segment['chord']}")
```

### Command Line Usage

```bash
# Basic chord recognition
python test_inference.py --audio_file path/to/audio.mp3

# Large vocabulary mode
python test_inference.py --audio_file path/to/audio.mp3 --vocabulary large_vocabulary

# Specify output directory
python test_inference.py --audio_file path/to/audio.mp3 --output_dir results/
```

## API Reference

### BTCChordRecognizer

#### `__init__(vocabulary_type='major_minor', config_path=None)`

Initialize the chord recognizer.

**Parameters:**
- `vocabulary_type` (str): Either `'major_minor'` (25 chord classes) or `'large_vocabulary'` (170 chord classes)
- `config_path` (str, optional): Path to configuration file. Uses default if None.

#### `recognize_chords(audio_path, save_results=True, output_dir=None)`

Recognize chords from a single audio file.

**Parameters:**
- `audio_path` (str): Path to audio file (mp3 or wav)
- `save_results` (bool): Whether to save results to files
- `output_dir` (str, optional): Output directory. Uses audio file directory if None.

**Returns:**
- `dict`: Dictionary containing:
  - `audio_file`: Path to processed audio file
  - `vocabulary_type`: Vocabulary type used
  - `song_length`: Length of song in seconds
  - `chord_segments`: List of chord segments with start/end times and chord names

#### `recognize_waveform(waveform, sample_rate, save_results=True, output_dir=None, audio_name="waveform")`

Recognize chords from raw audio waveform data.

**Parameters:**
- `waveform` (np.ndarray): Raw audio waveform (1D numpy array)
- `sample_rate` (int): Sample rate of the waveform in Hz
- `save_results` (bool): Whether to save results to files
- `output_dir` (str, optional): Output directory for results
- `audio_name` (str): Name for saved files (default: "waveform")

**Returns:**
- `dict`: Dictionary containing:
  - `audio_file`: Description of the audio source
  - `vocabulary_type`: Vocabulary type used
  - `song_length`: Length of song in seconds
  - `sample_rate`: Sample rate of the waveform
  - `num_samples`: Number of samples in the waveform
  - `chord_segments`: List of chord segments with start/end times and chord names

#### `recognize_batch(audio_dir, output_dir=None, file_extensions=('.mp3', '.wav'))`

Recognize chords for all audio files in a directory.

**Parameters:**
- `audio_dir` (str): Directory containing audio files
- `output_dir` (str, optional): Output directory for results
- `file_extensions` (tuple): Audio file extensions to process

**Returns:**
- `list`: List of result dictionaries (one per audio file)

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
    'audio_file': 'path/to/audio.mp3',
    'vocabulary_type': 'major_minor',
    'song_length': 257.22,
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

- MP3
- WAV
- Other formats supported by librosa

## Waveform API vs File API

### Use File API (`recognize_chords`) when:
- Processing existing audio files
- Batch processing multiple files
- Simple file-based workflows

### Use Waveform API (`recognize_waveform`) when:
- Processing audio from microphones or real-time streams
- Working with audio data in memory
- Building real-time applications
- Processing audio from non-file sources (databases, APIs, etc.)
- Need to avoid file I/O overhead
- Working with synthetic or generated audio

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

### Single File Processing
```python
from btc_chord_inference import BTCChordRecognizer

recognizer = BTCChordRecognizer()
results = recognizer.recognize_chords('song.mp3')

# Print chord sequence
for segment in results['chord_segments']:
    print(f"{segment['start_time']:.1f}s: {segment['chord']}")
```

### Waveform Processing
```python
import librosa
import numpy as np

recognizer = BTCChordRecognizer()

# Load audio as waveform
waveform, sample_rate = librosa.load('song.mp3', sr=None, mono=True)

# Recognize chords from waveform
results = recognizer.recognize_waveform(waveform, sample_rate)

# Process synthetic audio
duration = 10  # seconds
sample_rate = 22050
t = np.linspace(0, duration, int(sample_rate * duration), False)
synthetic_waveform = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

results = recognizer.recognize_waveform(synthetic_waveform, sample_rate)
```

### Batch Processing
```python
recognizer = BTCChordRecognizer()
results = recognizer.recognize_batch('audio_folder/', 'output_folder/')

for result in results:
    print(f"Processed: {result['audio_file']}")
    print(f"Chords: {len(result['chord_segments'])} segments")
```

### Custom Configuration
```python
recognizer = BTCChordRecognizer(
    vocabulary_type='large_vocabulary',
    config_path='custom_config.yaml'
)
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
