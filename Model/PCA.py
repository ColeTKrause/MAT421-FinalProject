from MLP import preprocess_from_data, Net, train, evaluate
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
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

data_scaled = StandardScaler().fit_transform(data)

pca = PCA(n_components=30)
pca_features = pca.fit_transform(data_scaled)

plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )

plt.xlabel('PCA Feature')
plt.ylabel('Explained variance')
plt.title('Feature Explained Variance')
plt.show()

# Plot PCA transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Data after PCA')

ax1.scatter(pca_features[:, 0], pca_features[:, 1], c='b', marker='o')
ax1.set_xlabel('PCA1')
ax1.set_ylabel('PCA2')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(pca_features[:, 0], pca_features[:, 1], pca_features[:, 2], c='b', marker='o')
ax2.set_xlabel('PCA1')
ax2.set_ylabel('PCA2')
ax2.set_zlabel('PCA3', rotation=90)

plt.show()

train_x, train_y, test_x, test_y = preprocess_from_data(pca_features[:,:4], y)

print(train_x)

trained_model = train(Net(4), train_x, train_y)

performance_metrics = evaluate(trained_model, test_x, test_y)