from data import *
from model import *
from torch.utils.data import DataLoader
import torch 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SPEC_DIR_TRAIN = '/home/clem3nti/projects/scribe/data/spectrograms_train'
ROLL_DIR_TRAIN = '/home/clem3nti/projects/scribe/data/piano_rolls_train'
SPEC_DIR_TEST = '/home/clem3nti/projects/scribe/data/spectrograms_test'
ROLL_DIR_TEST = '/home/clem3nti/projects/scribe/data/piano_rolls_test'

EPOCHS = 60
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
DEVICE = 'cpu'
FREQ_BINS = 96
print("constants initialized")

train_loader = DataLoader(SpectralData(SPEC_DIR_TRAIN, ROLL_DIR_TRAIN), batch_size=1, shuffle=True)
test_loader = DataLoader(SpectralData(SPEC_DIR_TEST, ROLL_DIR_TEST), batch_size=1, shuffle=False)
print("data loaders made")

model = AudioToMidi(FREQ_BINS)
model.to(DEVICE)
print("model loaded")

loss_fn = nn.BCELoss()
print("optimizer function loaded")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print("optimizer loaded")

print("beginning training loop")
for e in range(EPOCHS): 
    i=0
    for spec_batch, roll_batch in train_loader: 
        spec = spec_batch.to(DEVICE)
        roll = roll_batch.to(DEVICE)
        print(f"batch {i} loaded")

        preds = model(spec)
        loss = loss_fn(preds, roll)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss = {loss.item():.4f}")
        del spec, roll, preds, loss 
        torch.cuda.empty_cache()
        print(f"Batch: {i} of Epoch: {e} completed")
        i+=1
        
    print(f"Epoch {e} completed") 


print("eval loop beginning")
model.eval()
all_scores = []
with torch.no_grad():
    for spec_batch, roll_batch in test_loader: 
        spec = spec_batch.to(DEVICE)
        roll = roll_batch.to(DEVICE)

        preds = model(spec)
        bin_preds = (preds > 0.5).float()

        y_pred = preds.view(-1).cpu().numpy()
        y_true = roll.view(-1).cpu().numpy()
        
        cur_scores = dict() 
        cur_scores["acc"] = accuracy_score(y_true, y_pred) 
        cur_scores["prec"] = precision_score(y_true, y_pred, zero_division=0)
        cur_scores["rec"] = recall_score(y_true, y_pred, zero_division=0)
        cur_scores["f1"] = f1_score(y_true, y_pred, zero_division=0)
        all_scores.append(cur_scores)

i = 0
for score in all_scores: 
    print(f"Score: {i}, Metrics: {all_scores[i]}")
    i+=1
    
torch.save(model.state_dict(), "model_states/model1.pth")
print("model saved")

