from __future__ import absolute_import
from __future__ import print_function

import multiprocessing
from os import path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from cv2 import imread
from geffnet import create_model
from pandas import DataFrame
from pandas import concat
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.notebook import tqdm

kernel_type = '9c_b7_1e_640_ext_15ep'
image_size = 640
use_amp = False
enet_type = 'efficientnet-b7'
batch_size = 32
num_workers = multiprocessing.cpu_count()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_dim = 9
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'dcm'}

CLASS_NAMES = {0: 'AK',
               1: 'BCC',
               2: 'BKL',
               3: 'DF',
               4: 'SCC',
               5: 'VASC',
               6: 'melanoma',
               7: 'nevus',
               8: 'unknown'}

use_meta = False
use_external = '_ext' in kernel_type
mel_idx = 6

transforms_val = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize()
])


class SIIMISICDataset(Dataset):
    def __init__(self, csv, split, mode, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image = imread(row.filepath)
        image = image[:, :, ::-1]

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.mode == 'test':
            return torch.tensor(image).float()
        else:
            return torch.tensor(image).float(), torch.tensor(self.csv.iloc[index].target).long()


class enetv2(nn.Module):
    """Model"""

    def __init__(self, backbone, out_dim, n_meta_features=0, load_pretrained=False):
        super(enetv2, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = create_model(enet_type.replace('-', '_'), pretrained=load_pretrained)
        self.dropout = nn.Dropout(0.5)

        in_ch = self.enet.classifier.in_features
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        x = self.myfc(self.dropout(x))
        return x


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, n_test=1, get_output=False):
    """Validation Function"""
    model.eval()
    LOGITS = []
    PROBS = []
    with torch.no_grad():
        for data in tqdm(loader):
            if use_meta:
                data, meta = data
            #                 data, meta, target = data.to(device), meta.to(device), target.to(device)
            #                 logits = torch.zeros((data.shape[0], out_dim)).to(device)
            #                 probs = torch.zeros((data.shape[0], out_dim)).to(device)
            #                 for I in range(n_test):
            #                     l = model(get_trans(data, I), meta)
            #                     logits += l
            #                     probs += l.softmax(1)
            else:
                data = data.to(device)
                logits = torch.zeros((data.shape[0], out_dim)).to(device)
                probs = torch.zeros((data.shape[0], out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())

    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()

    if get_output:
        return LOGITS, PROBS
    else:
        return None


def predict_melanoma(image_locs, model_dir=''):
    dfs_split = []
    LOGITS = []
    df_val = DataFrame(image_locs, columns=['filepath'])

    for fold in range(5):  # not sampling different data in each fold so just one fold.
        dfs = []
        dataset_valid = SIIMISICDataset(df_val, 'train', mode='test', transform=transforms_val)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, num_workers=num_workers)

        model = enetv2(enet_type, n_meta_features=0, out_dim=out_dim)
        model = model.to(device)
        model_file = path.join(model_dir, f'{kernel_type}_best_o_fold{fold}.pth')
        state_dict = torch.load(model_file)
        state_dict = {k.replace('module.', ''): state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        this_LOGITS, this_PROBS = val_epoch(model, valid_loader, n_test=8, get_output=True)
        #         PROBS.append(this_PROBS)
        LOGITS.append(this_LOGITS)
        dfs.append(df_val)

        dfs = concat(dfs)
        dfs['pred'] = np.concatenate([this_PROBS]).squeeze()[:, mel_idx]
        dfs_split.append(dfs)
    return dfs_split, LOGITS


def ensemble(dfs_split, LOGITS, len=0):
    """Doing ensembling"""
    single_df = None
    preds_long = [0 for i in range(len)]
    for d in dfs_split:
        if single_df is None:
            single_df = d
        for i, d_ in enumerate(d['pred']):
            preds_long[i] += d_

    preds_long = [i / len for i in preds_long]
    preds_long = DataFrame(preds_long, columns=['pred'])
    single_df = concat([single_df['filepath'], preds_long], axis=1)
    return single_df, np.mean(LOGITS, axis=0)
