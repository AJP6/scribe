import os
import librosa
import pretty_midi
import numpy as np 

SAMPLE_RATE = 44100
HOP_LENGTH = 64

def wav_to_spec(audio_dir): 
    spec_path = '/home/clem3nti/projects/scribe/data/spectrograms'
    if not os.path.exists(spec_path): 
        os.makedirs(spec_path) 

    #sample parameters 
    n_bins = 96 #total freq bins
    bins_per_octave = 12 #num bins in each oct
    
    i = 0 
    for f in os.listdir(audio_dir): 
        if not f.endswith('.wav'):
                continue
        sample_name = "np_spec" + str(i) + ".npy"

        #y is numpy array containing wav file
        y, sr = librosa.load(os.path.join(audio_dir, f), sr=SAMPLE_RATE)

        C = librosa.cqt(y, sr=SAMPLE_RATE, 
                        hop_length=HOP_LENGTH, 
                        n_bins=n_bins, 
                        bins_per_octave=bins_per_octave)

        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        C_db = C_db + 80 #add 80 element wise to numpy array

        np.save(os.path.join(spec_path, sample_name), C_db)
        i+=1

def midi_to_roll(midi_dir): 
    note_path = '/home/clem3nti/projects/scribe/data/piano_rolls'
    if not os.path.exists(note_path): 
        os.makedirs(note_path) 

    i = 0
    for f in os.listdir(midi_dir): 
        if not f.endswith('.midi'):
            continue
        midi = pretty_midi.PrettyMIDI(os.path.join(midi_dir, f))
        midi = midi.get_piano_roll(fs=SAMPLE_RATE/HOP_LENGTH)
        midi = (midi > 0).astype(int)
        
        roll_name = 'proll' + str(i) + '.npy'
        np.save(os.path.join(note_path, roll_name), midi)
        i+=1
     
def main(): 
    audio_dir = '/home/clem3nti/projects/scribe/data/audio'
    midi_dir = '/home/clem3nti/projects/scribe/data/midi'
    wav_to_spec(audio_dir)
    midi_to_roll(midi_dir)

if __name__ == "__main__": 
    main()  
