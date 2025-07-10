import argparse
import os 
import pretty_midi
import librosa
import torch
from model.py import * 

FREQ_BINS = 768

def parse_args():
    parser = argparse.ArgumentParser(description="WAV to MIDI converter")
    parser.add_argument("wav_path", type=str, help="Path to the input WAV file")
    args = parser.parse_args()

    return args.wav_path

def init_model(save_path): 
    model = AudioToMidi

def load_wav(file_name): 
    pass

def convert_midi(piano_roll): 
    pass 

def main(): 
    #init model 
    model = init_model()

    #load file 
    spec_array = load_wav(parse_args)

    #transcribe
    piano_roll = model(spec_array)

    #convert to midi
    midi = convert_midi(piano_roll)

    #write out midi file

if __name__ == "__main__": 
    main()
