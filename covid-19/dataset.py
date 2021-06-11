from torch.utils.data.dataset import Dataset


class ImageDataset(Dataset):
    def __init__(self):
        super(ImageDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
