import argparse
import os 
import pretty_midi
import librosa
import torch
from model import * 
import numpy as np

FREQ_BINS = 96
SAMPLE_RATE = 44100
HOP_LENGTH = 64

USE_DEFAULT_INPUT_DIR = False  # we can set to false if we want user to enter full paths
DEFAULT_INPUT_DIR = '/home/clem3nti/projects/scribe/io/input'

def parse_args():
    parser = argparse.ArgumentParser(description="WAV to MIDI converter")
    parser.add_argument("wav_path", type=str, help="Path to the input WAV file")
    args = parser.parse_args()

    return args.wav_path

def init_model(): 
    model = AudioToMidi(input_freq_bins=FREQ_BINS)
    state_dict = torch.load('model_states/model1.pth')
    model.load_state_dict(state_dict)
    return model

def load_wav(file_name): 
    if USE_DEFAULT_INPUT_DIR:
        file_path = os.path.join(DEFAULT_INPUT_DIR, file_name)
    else:
        file_path = file_name

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"wav file not found: {file_path}")

    # y is numpy array containing wav file
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    C = librosa.cqt(y, sr=SAMPLE_RATE, 
                    hop_length=HOP_LENGTH, 
                    n_bins=FREQ_BINS, 
                    bins_per_octave=12)

    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    C_db = C_db + 80 # add 80 element wise to numpy array

    return C_db

#takes in a torch.Tensor()
def convert_midi(piano_roll, output_file, threshold = 0.5, base_midi_pitch = 12, fs = SAMPLE_RATE // HOP_LENGTH): 
    if isinstance(piano_roll, torch.Tensor):
        numpy_roll = piano_roll.detach().cpu().numpy() 

    midi_file = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for pitch_idx in range(numpy_roll.shape[0]):
        pitch = base_midi_pitch + pitch_idx
        is_active = numpy_roll[pitch_idx] > threshold # changed this from numpy_roll[pitch] to [pitch_idx] cause otherwise i think it woudld go out of bounds

        if not np.any(is_active):
            continue

        changes = np.diff(is_active.astype(int))
        notes_on = np.where(changes == 1)[0]
        notes_off = np.where(changes == -1)[0] # np.where() with only a condition argument returns a tuple with 1 index

        if is_active[0]:
            notes_on = np.insert(notes_on, 0, 0)
        if is_active[-1]:
            notes_off = np.append(notes_off, len(is_active) - 1)

        for on, off in zip(notes_on, notes_off):
            start = on / fs
            end = off / fs
            note = pretty_midi.Note(velocity = 100, pitch = pitch, start = start, end = end)
            instrument.notes.append(note)

    midi_file.instruments.append(instrument)
    midi_file.write(output_file)


def main(): 
    wav_path = parse_args()
    #init model 
    model = init_model()
    model.eval()
    #load file 
    spec_array = load_wav(wav_path)
    #transcribe

    with torch.no_grad():
        piano_roll = model(torch.tensor(spec_array).unsqueeze(0).unsqueeze(0).float())

    #convert to midi
    base = os.path.splitext(os.path.basename(wav_path))[0]
    output_file = base + ".midi"
    convert_midi(piano_roll.squeeze(0), output_file=output_file)

if __name__ == "__main__": 
    main()
