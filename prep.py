import os
import librosa
import pretty_midi
import numpy as np 

def wav_to_spec(audio_dir): 
    spec_path = '/home/clem3nti/projects/scribe/data/spectrograms'
    if not os.path.exists(spec_path): 
        os.makedirs(spec_path): 

    #sample parameters
    hop_length = 512
    n_bins = 96 #total freq bins
    bins_per_octave = 12 #num bins in each oct
    
    i = 0 
    for f in os.listdir(audio_dir): 
        sample_name = "np_spec" + str(i) + ".npy"

        #y is numpy array containing wav file
        y, sr = librosa.load(os.path.join(audio_dir, f), sr=None)

        C = librosa.cqt(y, sr=sr, 
                        hop_lenght=hop_length, 
                        n_bins=n_bins, 
                        bins_per_octave=bins_per_coctave)

        C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

        np.save(os.path.join(spec_path, sample_name), C_db)
        i+=1

def midi_to_roll(midi_dir): 
    note_path = '/home/clem3nti/projects/scribe/data/piano_rolls'
    if not os.path.exists(spec_path): 
        os.makedirs(spec_path): 

    i = 0
    for f in os.listdir(midi_dir): 
        midi = pretty_midi.PrettyMIDI(os.path.join(midi_dir, f))
        midi = midi.get_piano_roll(fs=512)
        midi = (midi > 0).astype(int)
        
        roll_name = 'proll' + str(i) + '.npy'
        np.save(os.path.join(note_path, roll_name), midi)
        i+=1
     
def main(): 
    audio_dir = '/home/clem3nti/porjects/scribe/data/audio'
    midi_dir = '/home/clem3nti/porjects/scribe/data/midi'
    audio_to_spec(audio_dir)
    midi_to_roll(midi_dir)

if __name__ == "__main__": 
    main()

