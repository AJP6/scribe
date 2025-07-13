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
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
FREQ_BINS = 96

train_loader = DataLoader(SpectralData(SPEC_DIR_TRAIN, ROLL_DIR_TRAIN), batch_size=10, shuffle=True)
test_loader = DataLoader(SpectralData(SPEC_DIR_TEST, ROLL_DIR_TEST), batch_size=10, shuffle=False)
model = AudioToMidi(FREQ_BINS)
model.to(DEVICE)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for e in range(EPOCHS): 
    for spec_batch, roll_batch in train_loader: 
        spec = spec_batch.to(device)
        roll = roll_batch.to(device)

        preds = model(spec)
        bin_preds = (preds > 0.5).float()
        loss = loss_fn(bin_preds, roll_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {e} : Loss = {loss.item():.4f}")

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for spec_batch, roll_batch in test_loader: 
        spec = spec_batch.to(device)
        roll = roll_batch.to(device)

        preds = model(spec)
        bin_preds = (preds > 0.5).float()
        all_preds.append(bin_preds.cpu())
        all_labels.append(roll_batch.cpu())
        
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    y_pred = all_preds.view(-1).numpy()
    y_true = all_labels.view(-1).numpy()

    
    
    acc = accuracy_score(y_true, y_pred) 
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

torch.save(model.state_dict(), "model_states/model1.pth")

