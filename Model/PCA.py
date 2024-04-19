from MLP import preprocess_from_data, Net, train, evaluate
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
import matplotlib.pyplot as plt

torch.manual_seed(42)

df = pd.read_csv('./Data/data.csv').dropna(axis=1)
x = df.drop(['id', 'diagnosis'], axis=1)

encoder = LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])
y = df['diagnosis']

data = np.array(x)

X = data - np.mean(data, axis=0)

covariance_matrix = X.T @ X / X.shape[0]

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

transformed_data = X @ eigenvectors

# Plot PCA transformed data
fig = plt.figure(figsize=(10, 8))
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c='b', marker='o')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Data after PCA')
plt.show()

train_x, train_y, test_x, test_y = preprocess_from_data(transformed_data[:,:4], y)

print(train_x)

trained_model = train(Net(4), train_x, train_y)

performance_metrics = evaluate(trained_model, test_x, test_y)