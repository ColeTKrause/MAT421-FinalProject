from sklearn.metrics import f1_score, r2_score, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


torch.manual_seed(42)

class BreatCancerDataset(Dataset):
    def __init__(self, train_x, train_y):
        super(Dataset, self).__init__()
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(train_y)

    def __getitem__(self, idx):
        sample = torch.tensor(train_x[idx], dtype=torch.double)
        label = torch.tensor(train_y[idx], dtype=torch.int)
        return sample, label

# Four layer neural network with RELU between hidden layers and softmax on output
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(30, 64) 
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)
        self.double()

    def forward(self, x):
        out = self.dropout(F.relu(self.fc1(x)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.dropout(F.relu(self.fc3(out)))
        out = self.fc4(out)

        return out 
    
# TODO: 
# - Create dataloader and get train/test splits

df = pd.read_csv('./Data/data.csv').dropna(axis=1)

encoder = LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])

x = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']


# split into train and evaluation (8 : 2) using train_test_split from sklearn
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=123)


# now make x and y tensors, think about the shape of train_x, it should be (total_examples, sequence_lenth, feature_size)
# we wlll make sequence_length just 1 for simplicity, and you could use unsqueeze at dimension 1 to do this
# also when you create tensor, it needs to be float type since pytorch training does not take default type read using pandas
train_x = torch.tensor(np.array(train_x))
train_y = torch.tensor(np.array(train_y))
test_x = torch.tensor(np.array(test_x)) 
seq_len = train_x[0].shape[0] # it is actually just 1 as explained above

train_dataset = BreatCancerDataset(train_x, train_y)
test_dataset = BreatCancerDataset(test_x, test_y)
train_loader = DataLoader(train_dataset)
test_loader = DataLoader(test_dataset)

# instantiate your MLP model and move to device as in cnn section
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net().to(device)
# loss function is nn.MSELoss since it is regression task
criterion = torch.nn.CrossEntropyLoss()
# try Adam optimizer (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with learning rate 0.0001, feel free to use other optimizer
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

epoch_size = 20

# start training 
net.train()
for epoch in range(epoch_size): 

    loss = 0.0 # you can print out average loss per batch every certain batches

    for batch_idx, data in enumerate(train_loader):
        # get inputs and target values from dataloaders and move to device
        inputs, targets = data
        inputs = inputs.to(device, dtype=torch.double)  # Ensure inputs are double
        targets = targets.to(device, dtype=torch.long)  # Ensure targets are long for classification
        optimizer.zero_grad()
        outputs = net(inputs)
        i_loss = criterion(outputs, targets)  # Unsqueezing targets for MSE loss
        i_loss.backward()
        optimizer.step()

        loss += i_loss # add loss for current batch
        if batch_idx % 100 == 99:    # print average loss per batch every 100 batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0

    scheduler.step(loss)

print('Finished Training')



predictions = []
ground_truth = []
# evaluation
net.eval()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        
        outputs = net(inputs)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Convert probabilities to binary predictions using a threshold of 0.5
        binary_predictions = (probabilities[:, 1] >= 0.5).int()
        
        predictions.extend(binary_predictions.tolist())
        ground_truth.extend(labels.tolist())

# Calculate evaluation metrics
accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions)
recall = recall_score(ground_truth, predictions)
f1 = f1_score(ground_truth, predictions)
auc_roc = roc_auc_score(ground_truth, predictions)

print('Accuracy: ', accuracy)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)
print('AUC ROC: ', auc_roc)


