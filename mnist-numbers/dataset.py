from torch.utils.data.dataset import Dataset
import pandas as pd
import torch


class DigitDataset(Dataset):
    def __init__(self, csv_file, is_train=True, device="cpu"):
        super(DigitDataset, self).__init__()
        data = pd.read_csv(csv_file)
        self.is_train = is_train
        self.device = device
        if is_train:
            self.labels = data['label']
            data.drop('label', axis=1, inplace=True)
        self.images = data.values.reshape(-1, 28, 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.is_train:
            return torch.tensor(image, dtype=torch.float32).to(self.device), torch.tensor(self.labels.iloc[0]).to(self.device)
        return torch.tensor(image, dtype=torch.float32).to(self.device)
