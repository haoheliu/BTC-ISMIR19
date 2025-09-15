"""
BTC Chord Recognition Inference Package
=====================================

A clean, reusable package for chord recognition using the Bi-directional Transformer for Chord Recognition (BTC) model.

Usage:
    from btc_chord_inference import BTCChordRecognizer
    
    recognizer = BTCChordRecognizer(vocabulary_type='major_minor')
    chords = recognizer.recognize_chords('path/to/audio.mp3')
"""

from .inference import BTCChordRecognizer

__version__ = "1.0.0"
__author__ = "BTC-ISMIR19"
__all__ = ["BTCChordRecognizer"]
