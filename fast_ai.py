import subprocess
import sys
import fastai
import timm
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import torch

from fastai.vision.all import *
from sklearn.model_selection import StratifiedGroupKFold

# Configuration class
class CFG:
    # File paths
    BASE_PATH = Path('./data')
    TRAIN_FEATURES_PATH = BASE_PATH / 'train_features.csv'
    TRAIN_LABELS_PATH = BASE_PATH / 'train_labels.csv'

    MODEL_ARCHITECTURE = 'timm/convnext_large.fb_in22k'
    IMAGE_SIZE = 512
    BATCH_SIZE = 32

    EPOCHS = 15

    # optimization
    NUM_WORKERS = 12
    PIN_MEMORY = True
    PREFETCH_FACTOR = 4

    TARGET_COL = 'label'
    SEED = 42
    BASE_LR = 1e-3

print(f"Configuration:")
print(f"   Model: {CFG.MODEL_ARCHITECTURE}")
print(f"   Resolution: {CFG.IMAGE_SIZE}x{CFG.IMAGE_SIZE}")
print(f"   Batch Size: {CFG.BATCH_SIZE}")
print(f"   Training Epochs: {CFG.EPOCHS}")

# Set random seed for reproducibility
set_seed(CFG.SEED, reproducible=True)

# Data preparation
print("\nPreparing data...")

import data

df = data.train_all


print(f"Data loaded successfully!")
print(f"   Training images: {len(df)}")
print(f"   Number of classes: {df['label'].nunique()}")

# Check class distribution
print("\nClass distribution:")
print(df['label'].value_counts())

print("Fold distribution:")
print(df.fold.value_counts())

# Create fold splitter function
def get_splitter(fold_num):
    def _inner(o):
        val_mask = o['fold'] == fold_num
        train_mask = o['fold'] != fold_num
        return o.index[train_mask], o.index[val_mask]
    return _inner

# DataBlock configuration
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader('filepath'),
    get_y=ColReader(CFG.TARGET_COL),
    splitter=get_splitter(0),
    item_tfms=Resize(CFG.IMAGE_SIZE, method=ResizeMethod.Pad, pad_mode=PadMode.Zeros),
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]
)

print(f"Creating DataLoaders (batch size: {CFG.BATCH_SIZE})...")

dls = dblock.dataloaders(
    df,
    bs=CFG.BATCH_SIZE,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.PIN_MEMORY,
    prefetch_factor=CFG.PREFETCH_FACTOR
)

print(f"Creating {CFG.MODEL_ARCHITECTURE} model...")

# Create learner with mixed precision
learn = vision_learner(dls, CFG.MODEL_ARCHITECTURE, metrics=[accuracy]).to_fp16()

print(f"Starting training for {CFG.EPOCHS} epochs...")

# Train the model
learn.fit_one_cycle(CFG.EPOCHS, lr_max=CFG.BASE_LR)
