import numpy as np

import torch
from torch.utils.data import Dataset

from monai.networks.nets import UNet
from monai.losses import PerceptualLoss
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import DiffusionInferer

from monai.utils import set_determinism

set_determinism(0)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

##################################################
# Dataset - direct npy load & slice extraction
##################################################
class ADNISliceDataset(Dataset):
    def __init__(self, npy_path, labels_path):
        a = np.load(npy_path, mmap_mode='r')  # [N, W, D, H, C]
        labels = np.load(labels_path)

        # drop last channel dim & reorder to [N, D, H, W]
        a = np.squeeze(a, axis=-1)
        a = np.transpose(a, (0, 2, 3, 1))

        # pick middle slice: a[:,57,:,:] -> [N, H, W]e
        self.data = np.expand_dims(a[:, 57, :, :], axis=1)  # -> [N,1,H,W]
        self.percentile = np.percentile(self.data, 99.5).astype(np.float32)
        self.data = np.clip(self.data / self.percentile, 0.0, 1.0)

        # convert labels to 3 classes: 0 (CN), 1 (MCI: pMCI+sMCI), 3 (AD)
        self.labels = np.where((labels == 1) | (labels == 2), 1, labels)
        self.labels = np.where(self.labels == 3, 2, self.labels)


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'image': image, 'label': label}


# === Dataset with demographic conditioning ===
class ADNISliceDemographicDataset(Dataset):
    def __init__(self, npy_path, labels_path, demographic_path):
        a = np.load(npy_path, mmap_mode='r')  # [N, W, D, H, C]
        labels = np.load(labels_path)
        demographics = np.load(demographic_path).astype(np.float32)  # [N, 5]

        a = np.squeeze(a, axis=-1)
        a = np.transpose(a, (0, 2, 3, 1))  # -> [N, D, H, W]

        self.data = np.expand_dims(a[:, 57, :, :], axis=1)  # -> [N, 1, H, W]
        self.percentile = np.percentile(self.data, 99.5).astype(np.float32)
        self.data = np.clip(self.data / self.percentile, 0.0, 1.0)

        self.labels = np.where((labels == 1) | (labels == 2), 1, labels)
        self.labels = np.where(self.labels == 3, 2, self.labels)

        # Normalize Age, MMSE, ADAS-Cog
        demo = demographics.copy()
        demo[:, 0] = (demo[:, 0] - 60.0) / (90.0 - 60.0)         # Age
        demo[:, 3] = (demo[:, 3] - 0.0) / (30.0 - 0.0)           # MMSE
        demo[:, 4] = (demo[:, 4] - 0.0) / (70.0 - 0.0)           # ADAS-Cog
        self.demographics = torch.tensor(demo, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        demo = self.demographics[idx]  # shape: [5]
        return {'image': image, 'label': label, 'demo': demo}
