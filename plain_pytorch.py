from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from fastai.torch_core import set_seed
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import itertools

import data
from dataset import ImageDatasetWithLabel, resize_to_square_with_reflect, resize_and_pad_square
from model import MySimpleModel


norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm)

def norm_bias_params(module, with_bias=True):
    if isinstance(module, norm_types): return list(module.parameters())

    res = list(itertools.chain(*list(map(
        lambda m: norm_bias_params(m, with_bias=with_bias),
        module.children()
    ))))

    if with_bias and getattr(module, 'bias', None) is not None: res.append(module.bias)

    return res



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

set_seed(CFG.SEED, reproducible=True)

model_preprocessor = v2.Compose([
    lambda input:
        # resize_to_square_with_reflect(input, size=CFG.IMAGE_SIZE, as_tensor=True),
        resize_and_pad_square(input, CFG.IMAGE_SIZE),
    v2.ToTensor(),

    v2.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
])

def get_training_data_loader(data, labels, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS):
    return DataLoader(
        ImageDatasetWithLabel(data=data, labels=labels, processor=model_preprocessor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

model = MySimpleModel(CFG.MODEL_ARCHITECTURE, num_classes=len(data.species_labels))

model = model.to('cuda')

scaler = torch.amp.GradScaler('cuda')

optimizer = torch.optim.AdamW([
    {'name': "encoder", "params": [],  "lr": 1e-4, "weight_decay": 0.05},
    {'name': "classifier", "params": [], "lr": 1e-3, "weight_decay": 0.01}
])

# Freezing
head_lr = CFG.BASE_LR

# freeze pretrained model
for p in model.parameters():
    p.requires_grad = False

# unfreeze head
for p in model.dict['head'].parameters():
    p.requires_grad = True

pretrained_bias_and_norm = []

# unfreeze bias and normalization parameters
for p in norm_bias_params(model.dict['body'], with_bias=False):
    p.requires_grad = True
    pretrained_bias_and_norm.append(p)

head_params = set(model.dict['head'].parameters())
head_params_without_decay = set(norm_bias_params(model.dict['head'], with_bias=True))

head_params_with_decay = list(head_params - head_params_without_decay)
head_params_without_decay = list(head_params_without_decay)

optimizer.param_groups = []

optimizer.add_param_group({'name': "head_with_decay", "params": head_params_with_decay, "lr": head_lr, "weight_decay": 0.01})
optimizer.add_param_group({'name': "head_without_decay", "params": head_params_without_decay, "lr": head_lr, "weight_decay": 0.0})
optimizer.add_param_group({'name': "pretrained_bias_and_norm", "params": pretrained_bias_and_norm, "lr": head_lr, "weight_decay": 0.0})

# Loss
criterion = nn.CrossEntropyLoss()

def sched_cos(start, end, pos):
    return start + (1 + np.cos(np.pi * (1 - pos))) * (end - start) / 2

def set_lr(optimizer, dataloader, max_epochs, epoch_idx, batch_idx):
    total_steps = max_epochs * len(dataloader.dataset)
    current_step = epoch_idx * len(dataloader.dataset) + batch_idx * dataloader.batch_size

    pos = current_step / total_steps

    lrs = 1e-3

    start = lrs / 25
    middle = lrs
    end = lrs / 1e5

    start_mom = 0.95
    middle_mom = 0.85
    end_mom = 0.95

    pct = 0.25

    if pos <= pct:
        lr = sched_cos(start, middle, pos / pct)
        mom = sched_cos(start_mom, middle_mom, pos / pct)
    else:
        lr = sched_cos(middle, end, (pos - pct) / (1 - pct))
        mom = sched_cos(middle_mom, end_mom, (pos - pct) / (1 - pct))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['betas'] = (mom, param_group['betas'][1])


fold_0_mask = data.train_all['fold'] == 0
fold_0_train = data.train_all[ ~fold_0_mask ] #.sample(n=100)
fold_0_valid = data.train_all[ fold_0_mask ] #.sample(n=100)

dataloader_train = get_training_data_loader(fold_0_train['filepath'], fold_0_train['label'])
dataloader_val = get_training_data_loader(fold_0_valid['filepath'], fold_0_valid['label'], shuffle=False)

tracking_loss = []

for cur_epoch in range(CFG.EPOCHS):
    print(f"Starting epoch {cur_epoch}")
    print(f"===============" + "=" * len(str(cur_epoch)))

    # TRAINING
    model.train()

    loss_acc = 0
    count = 0

    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc='Training'):
        optimizer.zero_grad(set_to_none=True)

        set_lr(optimizer, dataloader_train, CFG.EPOCHS, cur_epoch, batch_idx)

        images, labels = batch["inputs"].to("cuda"), batch["labels"].to("cuda")

        # out = model(images)
        # loss = criterion(out.logits, labels)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            out = model(images)
            loss = criterion(out.logits, labels)

        c = batch['inputs'].size(0)
        loss_acc += loss.item() * c
        count += c

        # loss.backward()
        # optimizer.step()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


    tracking_loss.append(loss_acc / count)

    print(f"Training loss: {(loss_acc / count):.4f}")

    # VALIDATION
    model.eval()

    loss_acc = 0
    count = 0

    preds_collector = []

    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for batch in tqdm.tqdm(dataloader_val, total=len(dataloader_val), desc='Validation'):
            logits = model.forward(batch["inputs"].to("cuda")).logits

            loss = criterion(logits, batch["labels"].to('cuda'))

            c = batch['inputs'].size(0)
            loss_acc += loss.item() * c
            count += c

            preds = F.softmax(logits, dim=1)

            preds_df = pd.DataFrame(
                preds.detach().to('cpu').numpy(),
                columns=data.species_labels,
            )
            preds_collector.append(preds_df)

    preds = pd.concat(preds_collector)

    print(f"Validation loss: {(loss_acc / count):.4f}")

    accuracy = (preds.to_numpy().argmax(axis=1) == fold_0_valid['label']).mean()

    print(f"Validation accuracy: {accuracy.item():.4f}")
