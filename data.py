from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

__all__ = [
    'train_all',
    'species_labels'
]

data_dir = Path(__file__).parent / "data/"

train_features = pd.read_csv(data_dir / "train_features.csv")
test_features = pd.read_csv(data_dir / "test_features.csv")
train_labels = pd.read_csv(data_dir / "train_labels.csv")

# verify one-hot encoding is correct
assert train_labels[train_labels.to_numpy()[:, 1:].sum(axis=1) != 1].shape[0] == 0

species_labels = sorted(train_labels.columns.unique())

train_features['filepath'] = str(data_dir) + '/' + train_features['filepath']
test_features['filepath'] = str(data_dir) + '/' + test_features['filepath']

train_labels['label'] = train_labels.to_numpy()[:, 1:].argmax(axis=1)

train_all = train_features.merge(train_labels[['id', 'label']], on='id')

train_all['fold'] = -1

splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(splitter.split(train_all, train_all['label'], groups=train_all['site'])):
    train_all.iloc[val_idx, train_all.columns.get_loc('fold')] = fold
