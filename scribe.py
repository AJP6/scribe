import argparse
import os 
import pretty_midi
import librosa
import torch
from model.py import * 

FREQ_BINS = 96
SAMPLE_RATE = 44100
HOP_LENGTH = 64

def parse_args():
    parser = argparse.ArgumentParser(description="WAV to MIDI converter")
    parser.add_argument("wav_path", type=str, help="Path to the input WAV file")
    args = parser.parse_args()

    return args.wav_path

def init_model(): 
    model = AudioToMidi
    state_dict = torch.load('model_states/model1.pth')
    model.load_state_dict(state_dict)
    return model

def load_wav(file_name): 
    load_path = '/home/clem3nti/projects/scribe/io/input'

    n_bins = 96 #total freq bins
    bins_per_octave = 12 #num bins in each oct

    file_path = os.path.join(load_path, file_name)
    #y is numpy array containing wav file
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    C = librosa.cqt(y, sr=SAMPLE_RATE, 
                    hop_length=HOP_LENGTH, 
                    n_bins=n_bins, 
                    bins_per_octave=bins_per_octave)

    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    C_db = C_db + 80 #add 80 element wise to numpy array

    return C_db

#takes in a torch.Tensor()
def convert_midi(piano_roll): 
    numpy_roll = piano_roll.cpu().numpy() 
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    fs = 120

def main(): 
    #init model 
    model = init_model()
    #load file 
    spec_array = load_wav(parse_args)
    #transcribe
    piano_roll = model(spec_array)
    #convert to midi
    midi = convert_midi(piano_roll).squeeze(0)

if __name__ == "__main__": 
    main()
