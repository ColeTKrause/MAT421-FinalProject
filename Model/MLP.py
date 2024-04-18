from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
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

# Dataset
class BreatCancerDataset(Dataset):
    def __init__(self, data, labels):
        super(BreatCancerDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.double)
        label = torch.tensor(self.labels[idx], dtype=torch.int)
        return sample, label

# Neural Network 
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
    
class Metric():

    def __init__(self, accuracy, precision, recall, f1, roc_auc):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.roc_auc = roc_auc


# Data preprocessing
def preprocess(data_path):

    df = pd.read_csv(data_path).dropna(axis=1)

    encoder = LabelEncoder()
    df['diagnosis'] = encoder.fit_transform(df['diagnosis'])

    x = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']

    # TODO: Try normalizing the training data. How does it improve performance?

    # split into train and evaluation (8 : 2) using train_test_split from sklearn
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=123)

    # Convert the partitioned data into Tensors
    train_x = torch.tensor(np.array(train_x))
    train_y = torch.tensor(np.array(train_y))
    test_x = torch.tensor(np.array(test_x)) 
    test_y = torch.tensor(np.array(test_y))

    return train_x, train_y, test_x, test_y

def train(untrained_net, train_x, train_y):
    # Instantiate the Datasets and Loaders
    train_dataset = BreatCancerDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset)

    # instantiate your MLP model and move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = untrained_net.to(device)
    # loss function is nn.CrossEntropyLoss() for classification
    criterion = torch.nn.CrossEntropyLoss()
    # Use an Adam optimizer with a learing rate scheduler to prevent overfitting
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    # Number of epochs to use during training
    epoch_size = 15

    # start training 
    net.train()
    for epoch in range(epoch_size): 

        loss = 0.0 # running loss

        for batch_idx, data in enumerate(train_loader):
            # get inputs and target values from dataloaders and move to device
            inputs, targets = data
            inputs = inputs.to(device, dtype=torch.double)  # Ensure inputs are double
            targets = targets.to(device, dtype=torch.long)  # Ensure targets are long for classification
            optimizer.zero_grad()
            outputs = net(inputs)
            i_loss = criterion(outputs, targets)
            i_loss.backward()
            optimizer.step()

            loss += i_loss # add loss for current batch
            if batch_idx % 100 == 99:    # print average loss per batch every 100 batches
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {loss / 100:.3f}')
                loss = 0.0

        scheduler.step(loss)

    print('Finished Training')

    return net

# Start evaluation
def evaluate(trained_net, test_x, test_y):

    test_dataset = BreatCancerDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    predictions = []
    ground_truth = []
    trained_net.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            
            outputs = trained_net(inputs)
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

    return Metric(accuracy, precision, recall, f1, auc_roc)

# Demo/Usage example

train_x, train_y, test_x, test_y = preprocess('./Data/data.csv')

trained_net = train(Net(), train_x, train_y)

performance_metrics = evaluate(trained_net, test_x, test_y)