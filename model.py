import torch
import torch.nn as nn 
import torch.nn.functional as F

class AudioToMidi(nn.Module): 
    def __init__(self, input_freq_bins=96): 
        super().__init__()

        #convolution layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=1), 
            nn.BatchNorm2d(num_features=32), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(1,2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1), 
            nn.BatchNorm2d(num_features=64), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(1,2)), 

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1), 
            nn.BatchNorm2d(num_features=128), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(1,2))
        )

        #output layers 
        #in features in out_channels * freq_bins
        final_features = 128*input_freq_bins
        self.lin = nn.Linear(in_features=final_features, out_features=96)


    def forward(self, x):
        #[batch, channels, bins, time]
        x = self.conv_block(x)  
        #[batch, time, bins, channels]
        x = x.permute(0, 3, 2, 1)
        Ba, T, Bi, C = x.shape
        x = x.reshape(Ba, T, Bi*C)

        x = self.lin(x)
        x = x.permute(0, 2, 1)
        #returned tensor is of shape [freq, time]
        return torch.sigmoid(x)
        
