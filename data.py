import torch 
import os
import numpy as np
from torch.utils.data import Dataset 

MAX_LEN = 2048

def pad_crop(x): 
    if x.shape[-1] > MAX_LEN: 
        return x[..., :MAX_LEN]
    elif x.shape[-1] < MAX_LEN: 
        pad_width = MAX_LEN - x.shape[-1]
        return np.pad(x, ((0,0), (0, pad_width)), mode='constant')
    else: 
        return x


class SpectralData(Dataset): 
    def __init__(self, spec_dir, proll_dir): 
        self.spec_dir = spec_dir
        self.proll_dir = proll_dir
        self.pairs = []

        #list of spec and roll names paired 
        spec = sorted([f for f in os.listdir(spec_dir)])
        roll = sorted([f for f in os.listdir(proll_dir)])
        for s, r in zip(spec, roll): 
            self.pairs.append([s,r])

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx): 
        spec_path = os.path.join(self.spec_dir, self.pairs[idx][0])
        midi_path = os.path.join(self.proll_dir, self.pairs[idx][1])

        piano_roll = pad_crop(np.load(midi_path))
        spec = pad_crop(np.load(spec_path))

        #unsqueeze to add a dim for channels for the 
        #(bins, time_steps) --> (channels, bins, time_steps)
        #inner most dim is last
        #outer most dim is first
        t_spec = torch.tensor(spec).unsqueeze(0) 
        t_piano_roll = torch.tensor(piano_roll)

        #convert to float 32
        return t_spec.float(), t_piano_roll.float()
        

