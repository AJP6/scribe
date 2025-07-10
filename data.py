import torch 
import os
from torch.utils.data import Dataset 


class SpectralData(Dataset, spec_dir, proll_dir): 
    def __init__(self): 
        self.spec_dir = spec_dir
        self.proll_dir = proll_dir
        self.pairs = []

        #list of spec and roll names paired 
        spec = [f for f in os.listdir(spec_dir)]
        roll = [f for f in os.listdir(proll_dir)]
        for s, r in zip(spec, roll): 
            self.pairs.append([s,r])

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx): 
        spec_path = os.path.join(self.spec_dir, self.pairs[idx][0])
        midi_path = os.path.join(self.spec_dir, self.pairs[idx][1])

        piano_roll = np.load(midi_path)
        spec = np.load(spec_path)

        #unsqueeze to add a dim for channels for the 
        #(bins, time_steps) --> (channels, bins, time_steps)
        #inner most dim is last
        #outer most dim is first
        t_spec = torch.tensor(spec).unsqueeze(0) 
        t_piano_roll = torch.tensor(piano_roll)

        #convert to float 32
        return t_spec.float(), t_pianoroll.float()
        

