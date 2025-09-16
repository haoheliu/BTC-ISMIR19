import os
import torch
import numpy as np
import warnings
import librosa
from pathlib import Path

from models.btc_model import BTC_model
from utils.hparams import HParams
from utils.logger import info as logger_info, logging_verbosity
from utils.mir_eval_modules import (
    audio_file_to_features, 
    idx2chord, 
    idx2voca_chord, 
    get_audio_paths
)
import torch.nn as nn
warnings.filterwarnings('ignore')
logging_verbosity(1)

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model (nn.Module): The PyTorch model to inspect.
        trainable_only (bool): If True, count only parameters with requires_grad=True.
    
    Returns:
        int: Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def waveform_to_features(waveform, sample_rate, config):
    """
    Convert raw audio waveform to CQT features
    
    Args:
        waveform (np.ndarray): Raw audio waveform
        sample_rate (int): Sample rate of the waveform
        config: Configuration object containing feature parameters
        
    Returns:
        tuple: (feature, feature_per_second, song_length_second)
    """
    # Resample if necessary
    target_sr = config.mp3['song_hz']
    # Ensure sample_rate is a scalar integer
    if hasattr(sample_rate, '__len__') and len(sample_rate) > 0:
        sample_rate = int(sample_rate[0])
    else:
        sample_rate = int(sample_rate)
    
    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr
    
    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform)
    
    # Process in chunks
    currunt_sec_hz = 0
    inst_len_samples = int(config.mp3['song_hz'] * config.mp3['inst_len'])
    
    while len(waveform) > currunt_sec_hz + inst_len_samples:
        start_idx = int(currunt_sec_hz)
        end_idx = int(currunt_sec_hz + inst_len_samples)
        
        tmp = librosa.cqt(
            waveform[start_idx:end_idx], 
            sr=sample_rate, 
            n_bins=config.feature['n_bins'], 
            bins_per_octave=config.feature['bins_per_octave'], 
            hop_length=config.feature['hop_length']
        )
        
        if start_idx == 0:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)
        
        currunt_sec_hz = end_idx
    
    # Process remaining samples
    if currunt_sec_hz < len(waveform):
        tmp = librosa.cqt(
            waveform[currunt_sec_hz:], 
            sr=sample_rate, 
            n_bins=config.feature['n_bins'], 
            bins_per_octave=config.feature['bins_per_octave'], 
            hop_length=config.feature['hop_length']
        )
        if currunt_sec_hz == 0:
            feature = tmp
        else:
            feature = np.concatenate((feature, tmp), axis=1)
    
    # Apply log transformation
    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = config.mp3['inst_len'] / config.model['timestep']
    song_length_second = len(waveform) / config.mp3['song_hz']
    
    return feature, feature_per_second, song_length_second


class BTCChordRecognizer:
    """
    BTC Chord Recognition Class
    
    Provides a simple interface for chord recognition using the BTC model.
    """
    
    def __init__(self, vocabulary_type='major_minor', config_path=None):
        """
        Initialize the BTC Chord Recognizer
        
        Args:
            vocabulary_type (str): Either 'major_minor' or 'large_vocabulary'
            config_path (str): Path to config file. If None, uses default config.
        """
        self.vocabulary_type = vocabulary_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up paths
        package_dir = Path(__file__).parent
        if config_path is None:
            config_path = package_dir / "run_config.yaml"
        
        # Load configuration
        self.config = HParams.load(config_path)
        
        # Set up model based on vocabulary type
        if vocabulary_type == 'large_vocabulary':
            self.config.feature['large_voca'] = True
            self.config.model['num_chords'] = 170
            model_file = package_dir / "models" / "btc_model_large_voca.pt"
            self.idx_to_chord = idx2voca_chord()
        else:
            model_file = package_dir / "models" / "btc_model.pt"
            self.idx_to_chord = idx2chord
            
        # Initialize model
        self.model = BTC_model(config=self.config.model).to(self.device)
        
        # Load model weights
        if os.path.isfile(model_file):
            checkpoint = torch.load(model_file, map_location=self.device)
            self.mean = checkpoint['mean']
            self.std = checkpoint['std']
            self.model.load_state_dict(checkpoint['model'])
            logger_info(f"Model loaded: {vocabulary_type} vocabulary")
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")
    
    def _apply_chord_delay(self, chord_segments, delay_seconds=0.1):
        """
        Apply a delay to chord segment timestamps to compensate for early detection
        
        Args:
            chord_segments (list): List of chord segments
            delay_seconds (float): Delay to apply in seconds
            
        Returns:
            list: Updated chord segments with delayed timestamps
        """
        if not chord_segments:
            return chord_segments
            
        updated_segments = []
        for segment in chord_segments:
            # Apply delay to start_time, ensuring it doesn't go below 0
            new_start_time = max(0.0, segment['start_time'] + delay_seconds)
            new_end_time = segment['end_time'] + delay_seconds
            
            updated_segments.append({
                'start_time': new_start_time,
                'end_time': new_end_time,
                'chord': segment['chord']
            })
        
        return updated_segments
    
    def recognize_chords(self, audio_path, save_results=True, output_dir=None):
        """
        Recognize chords from an audio file
        
        Args:
            audio_path (str): Path to audio file (mp3 or wav)
            save_results (bool): Whether to save results to files
            output_dir (str): Directory to save results. If None, saves in same directory as audio
            
        Returns:
            dict: Dictionary containing chord recognition results
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger_info(f"Processing: {audio_path}")
        
        # Load and process audio
        feature, feature_per_second, song_length_second = audio_file_to_features(
            str(audio_path), self.config
        )
        
        # Prepare features
        feature = feature.T
        feature = (feature - self.mean) / self.std
        time_unit = feature_per_second
        n_timestep = self.config.model['timestep']
        
        # Pad features to match timestep
        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep
        
        # Perform inference
        start_time = 0.0
        chord_segments = []
        
        with torch.no_grad():
            self.model.eval()
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            for t in range(num_instance):
                self_attn_output, _ = self.model.self_attn_layers(
                    feature_tensor[:, n_timestep * t:n_timestep * (t + 1), :]
                )
                prediction, _ = self.model.output_layer(self_attn_output)
                prediction = prediction.squeeze()
                
                for i in range(n_timestep):
                    if t == 0 and i == 0:
                        prev_chord = prediction[i].item()
                        continue
                    if prediction[i].item() != prev_chord:
                        chord_segments.append({
                            'start_time': start_time,
                            'end_time': time_unit * (n_timestep * t + i),
                            'chord': self.idx_to_chord[prev_chord]
                        })
                        start_time = time_unit * (n_timestep * t + i)
                        prev_chord = prediction[i].item()
                    if t == num_instance - 1 and i + num_pad == n_timestep:
                        if start_time != time_unit * (n_timestep * t + i):
                            chord_segments.append({
                                'start_time': start_time,
                                'end_time': time_unit * (n_timestep * t + i),
                                'chord': self.idx_to_chord[prev_chord]
                            })
                        break
        
        # Apply 0.1 second delay to compensate for early detection
        # chord_segments = self._apply_chord_delay(chord_segments, delay_seconds=0.08)
        
        results = {
            'audio_file': str(audio_path),
            'vocabulary_type': self.vocabulary_type,
            'song_length': song_length_second,
            'chord_segments': chord_segments
        }
        
        # Save results if requested
        if save_results:
            if output_dir is None:
                output_dir = audio_path.parent
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            self._save_results(results, output_dir)
        
        return results
    
    def _save_results(self, results, output_dir):
        """Save chord recognition results to files"""
        audio_name = Path(results['audio_file']).stem
        
        # Save .lab file
        lab_file = output_dir / f"{audio_name}.lab"
        with open(lab_file, 'w') as f:
            for segment in results['chord_segments']:
                f.write(f"{segment['start_time']:.3f} {segment['end_time']:.3f} {segment['chord']}\n")
        
        logger_info(f"Chord labels saved: {lab_file}")
        
        # Save MIDI file
        try:
            import mir_eval
            import pretty_midi as pm
            
            starts, ends, pitchs = [], [], []
            intervals = [(s['start_time'], s['end_time']) for s in results['chord_segments']]
            chords = [s['chord'] for s in results['chord_segments']]
            
            for p in range(12):
                for i, (interval, chord) in enumerate(zip(intervals, chords)):
                    root_num, relative_bitmap, _ = mir_eval.chord.encode(chord)
                    tmp_label = mir_eval.chord.rotate_bitmap_to_root(relative_bitmap, root_num)[p]
                    if i == 0:
                        start_time = interval[0]
                        label = tmp_label
                        continue
                    if tmp_label != label:
                        if label == 1.0:
                            starts.append(start_time)
                            ends.append(interval[0])
                            pitchs.append(p + 48)
                        start_time = interval[0]
                        label = tmp_label
                    if i == (len(intervals) - 1): 
                        if label == 1.0:
                            starts.append(start_time)
                            ends.append(interval[1])
                            pitchs.append(p + 48)
            
            midi = pm.PrettyMIDI()
            instrument = pm.Instrument(program=0)
            
            for start, end, pitch in zip(starts, ends, pitchs):
                pm_note = pm.Note(velocity=120, pitch=pitch, start=start, end=end)
                instrument.notes.append(pm_note)
            
            midi.instruments.append(instrument)
            midi_file = output_dir / f"{audio_name}.midi"
            midi.write(str(midi_file))
            logger_info(f"MIDI file saved: {midi_file}")
            
        except ImportError:
            logger_info("mir_eval or pretty_midi not available. MIDI file not generated.")
    
    def recognize_waveform(self, waveform, sample_rate, save_results=False, output_dir=None, audio_name="waveform"):
        """
        Recognize chords from raw audio waveform
        
        Args:
            waveform (np.ndarray): Raw audio waveform (1D array)
            sample_rate (int): Sample rate of the waveform
            save_results (bool): Whether to save results to files
            output_dir (str, optional): Output directory for results
            audio_name (str): Name for saved files (default: "waveform")
            
        Returns:
            dict: Dictionary containing chord recognition results
        """
        logger_info(f"Processing waveform: {len(waveform)} samples at {sample_rate}Hz")
        
        # Convert waveform to features
        feature, feature_per_second, song_length_second = waveform_to_features(
            waveform, sample_rate, self.config
        )
        
        # Prepare features
        feature = feature.T
        feature = (feature - self.mean) / self.std
        time_unit = feature_per_second
        n_timestep = self.config.model['timestep']
        
        # Pad features to match timestep
        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep
        
        # Perform inference
        start_time = 0.0
        chord_segments = []
        
        with torch.no_grad():
            self.model.eval()
            feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            for t in range(num_instance):
                self_attn_output, _ = self.model.self_attn_layers(
                    feature_tensor[:, n_timestep * t:n_timestep * (t + 1), :]
                )
                prediction, _ = self.model.output_layer(self_attn_output)
                prediction = prediction.squeeze()
                
                for i in range(n_timestep):
                    if t == 0 and i == 0:
                        prev_chord = prediction[i].item()
                        continue
                    if prediction[i].item() != prev_chord:
                        chord_segments.append({
                            'start_time': start_time,
                            'end_time': time_unit * (n_timestep * t + i),
                            'chord': self.idx_to_chord[prev_chord]
                        })
                        start_time = time_unit * (n_timestep * t + i)
                        prev_chord = prediction[i].item()
                    if t == num_instance - 1 and i + num_pad == n_timestep:
                        if start_time != time_unit * (n_timestep * t + i):
                            chord_segments.append({
                                'start_time': start_time,
                                'end_time': time_unit * (n_timestep * t + i),
                                'chord': self.idx_to_chord[prev_chord]
                            })
                        break
        
        # Apply 0.1 second delay to compensate for early detection
        # chord_segments = self._apply_chord_delay(chord_segments, delay_seconds=0.1)
        
        results = {
            'audio_file': f"{audio_name} (waveform)",
            'vocabulary_type': self.vocabulary_type,
            'song_length': song_length_second,
            'sample_rate': sample_rate,
            'num_samples': len(waveform),
            'chord_segments': chord_segments
        }
        
        # Save results if requested
        if save_results:
            if output_dir is None:
                output_dir = Path.cwd()
            else:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            self._save_waveform_results(results, output_dir, audio_name)
        
        return results
    
    def _save_waveform_results(self, results, output_dir, audio_name):
        """Save waveform chord recognition results to files"""
        # Save .lab file
        lab_file = output_dir / f"{audio_name}.lab"
        with open(lab_file, 'w') as f:
            for segment in results['chord_segments']:
                f.write(f"{segment['start_time']:.3f} {segment['end_time']:.3f} {segment['chord']}\n")
        
        logger_info(f"Chord labels saved: {lab_file}")
        
        # Save MIDI file
        try:
            import mir_eval
            import pretty_midi as pm
            
            starts, ends, pitchs = [], [], []
            intervals = [(s['start_time'], s['end_time']) for s in results['chord_segments']]
            chords = [s['chord'] for s in results['chord_segments']]
            
            for p in range(12):
                for i, (interval, chord) in enumerate(zip(intervals, chords)):
                    root_num, relative_bitmap, _ = mir_eval.chord.encode(chord)
                    tmp_label = mir_eval.chord.rotate_bitmap_to_root(relative_bitmap, root_num)[p]
                    if i == 0:
                        start_time = interval[0]
                        label = tmp_label
                        continue
                    if tmp_label != label:
                        if label == 1.0:
                            starts.append(start_time)
                            ends.append(interval[0])
                            pitchs.append(p + 48)
                        start_time = interval[0]
                        label = tmp_label
                    if i == (len(intervals) - 1): 
                        if label == 1.0:
                            starts.append(start_time)
                            ends.append(interval[1])
                            pitchs.append(p + 48)
            
            midi = pm.PrettyMIDI()
            instrument = pm.Instrument(program=0)
            
            for start, end, pitch in zip(starts, ends, pitchs):
                pm_note = pm.Note(velocity=120, pitch=pitch, start=start, end=end)
                instrument.notes.append(pm_note)
            
            midi.instruments.append(instrument)
            midi_file = output_dir / f"{audio_name}.midi"
            midi.write(str(midi_file))
            logger_info(f"MIDI file saved: {midi_file}")
            
        except ImportError:
            logger_info("mir_eval or pretty_midi not available. MIDI file not generated.")
    
    def recognize_waveforms_batch(self, waveforms, sample_rates, audio_names=None, save_results=False, output_dir=None):
        """
        Recognize chords from multiple raw audio waveforms in batch
        
        Args:
            waveforms (list): List of raw audio waveforms (each is 1D numpy array)
            sample_rates (list): List of sample rates corresponding to each waveform
            audio_names (list, optional): List of names for each waveform
            save_results (bool): Whether to save results to files
            output_dir (str, optional): Output directory for results
            
        Returns:
            list: List of result dictionaries, one for each waveform
        """
        if len(waveforms) != len(sample_rates):
            raise ValueError("Number of waveforms must match number of sample rates")
        
        if audio_names is None:
            audio_names = [f"waveform_{i}" for i in range(len(waveforms))]
        elif len(audio_names) != len(waveforms):
            raise ValueError("Number of audio names must match number of waveforms")
        
        logger_info(f"Processing batch of {len(waveforms)} waveforms")
        
        # Process each waveform using the existing single waveform method
        # This ensures reliability and consistency
        results = []
        for i, (waveform, sample_rate, audio_name) in enumerate(zip(waveforms, sample_rates, audio_names)):
            logger_info(f"Processing waveform {i+1}/{len(waveforms)}: {audio_name}")
            result = self.recognize_waveform(
                waveform, 
                sample_rate, 
                save_results=save_results,
                output_dir=output_dir,
                audio_name=audio_name
            )
            results.append(result)
        
        logger_info(f"Batch processing completed: {len(results)} waveforms processed")
        return results
    
    def _group_waveforms_by_length(self, waveforms, sample_rates, audio_names):
        """
        Group waveforms by similar lengths for efficient batch processing
        """
        # Calculate duration for each waveform
        waveform_info = []
        for waveform, sample_rate, audio_name in zip(waveforms, sample_rates, audio_names):
            duration = len(waveform) / sample_rate
            waveform_info.append((waveform, sample_rate, audio_name, duration))
        
        # Sort by duration
        waveform_info.sort(key=lambda x: x[3])
        
        # Group by duration ranges (within 10% of each other)
        groups = []
        current_group = []
        
        for waveform, sample_rate, audio_name, duration in waveform_info:
            if not current_group:
                current_group = [(waveform, sample_rate, audio_name, duration)]
            else:
                # Check if this waveform can be grouped with current group
                avg_duration = sum(info[3] for info in current_group) / len(current_group)
                if abs(duration - avg_duration) / avg_duration < 0.1:  # Within 10%
                    current_group.append((waveform, sample_rate, audio_name, duration))
                else:
                    # Start new group
                    groups.append([(info[0], info[1], info[2]) for info in current_group])
                    current_group = [(waveform, sample_rate, audio_name, duration)]
        
        # Add the last group
        if current_group:
            groups.append([(info[0], info[1], info[2]) for info in current_group])
        
        return groups
    
    def _process_waveform_batch(self, waveforms, sample_rates, audio_names):
        """
        Process a batch of waveforms with similar lengths efficiently
        """
        results = []
        
        # Process each waveform in the batch
        for waveform, sample_rate, audio_name in zip(waveforms, sample_rates, audio_names):
            # Convert waveform to features
            feature, feature_per_second, song_length_second = waveform_to_features(
                waveform, sample_rate, self.config
            )
            
            # Prepare features
            feature = feature.T
            feature = (feature - self.mean) / self.std
            time_unit = feature_per_second
            n_timestep = self.config.model['timestep']
            
            # Pad features to match timestep
            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep
            
            # Perform inference
            start_time = 0.0
            chord_segments = []
            
            with torch.no_grad():
                self.model.eval()
                feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                for t in range(num_instance):
                    self_attn_output, _ = self.model.self_attn_layers(
                        feature_tensor[:, n_timestep * t:n_timestep * (t + 1), :]
                    )
                    prediction, _ = self.model.output_layer(self_attn_output)
                    prediction = prediction.squeeze()
                    
                    for i in range(n_timestep):
                        if t == 0 and i == 0:
                            prev_chord = prediction[i].item()
                            continue
                        if prediction[i].item() != prev_chord:
                            chord_segments.append({
                                'start_time': start_time,
                                'end_time': time_unit * (n_timestep * t + i),
                                'chord': self.idx_to_chord[prev_chord]
                            })
                            start_time = time_unit * (n_timestep * t + i)
                            prev_chord = prediction[i].item()
                        if t == num_instance - 1 and i + num_pad == n_timestep:
                            if start_time != time_unit * (n_timestep * t + i):
                                chord_segments.append({
                                    'start_time': start_time,
                                    'end_time': time_unit * (n_timestep * t + i),
                                    'chord': self.idx_to_chord[prev_chord]
                                })
                            break
            
            # Apply 0.1 second delay to compensate for early detection
            # chord_segments = self._apply_chord_delay(chord_segments, delay_seconds=0.1)
            
            result = {
                'audio_file': f"{audio_name} (waveform)",
                'vocabulary_type': self.vocabulary_type,
                'song_length': song_length_second,
                'sample_rate': sample_rate,
                'num_samples': len(waveform),
                'chord_segments': chord_segments
            }
            
            results.append(result)
        
        return results
    
    def recognize_batch(self, audio_dir, output_dir=None, file_extensions=('.mp3', '.wav')):
        """
        Recognize chords for all audio files in a directory
        
        Args:
            audio_dir (str): Directory containing audio files
            output_dir (str): Directory to save results
            file_extensions (tuple): Audio file extensions to process
            
        Returns:
            list: List of results dictionaries
        """
        audio_paths = []
        for ext in file_extensions:
            audio_paths.extend(Path(audio_dir).glob(f"*{ext}"))
        
        if not audio_paths:
            logger_info(f"No audio files found in {audio_dir}")
            return []
        
        results = []
        for i, audio_path in enumerate(audio_paths):
            logger_info(f"Processing {i+1}/{len(audio_paths)}: {audio_path.name}")
            try:
                result = self.recognize_chords(str(audio_path), save_results=True, output_dir=output_dir)
                results.append(result)
            except Exception as e:
                logger_info(f"Error processing {audio_path}: {e}")
        
        return results
